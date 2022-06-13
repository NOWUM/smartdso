from math import exp
import numpy as np
import pandas as pd
from datetime import datetime, timedelta as td
import logging
from pvlib.pvsystem import PVSystem
from matplotlib import pyplot as plt
from pyomo.environ import Constraint, Var, Objective, SolverFactory, ConcreteModel, \
    Reals, Binary, minimize, maximize, value, quicksum, ConstraintList, Piecewise

# example to Piecewise:
# http://yetanothermathprogrammingconsultant.blogspot.com/2019/02/piecewise-linear-functions-and.html
# http://yetanothermathprogrammingconsultant.blogspot.com/2015/10/piecewise-linear-functions-in-mip-models.html

from participants.basic import BasicParticipant
from participants.utils import Resident
from demLib.electric_profile import StandardLoadProfile

# -> set logging options
logging.basicConfig()
LOG_LEVEL = "INFO"
logger = logging.getLogger('residential')
logger.setLevel(LOG_LEVEL)

# -> price data from survey
MEAN_PRICE = 28.01
VAR_PRICE = 7.9

RESOLUTION = {1440: 'min', 96: '15min', 24: 'h'}

# -> default prices EPEX-SPOT 2015
TARIFF = pd.read_csv(r'./participants/data/default_prices.csv', index_col=0)
TARIFF = TARIFF / 10  # -> [€/MWh] in [ct/kWh]
TARIFF.index = pd.to_datetime(TARIFF.index)
CHARGES = {'others': 2.9, 'taxes': 8.0, 'eeg': 6.5}
for values in CHARGES.values():
    TARIFF += values


class HouseholdModel(BasicParticipant):

    def __init__(self, residents: int,
                 demandP: float, london_data: bool = False, l_id: str = None,
                 ev_ratio: float = 0.5,
                 pv_system: dict = None,
                 grid_node: str = None,
                 start_date: datetime = datetime(2022, 1, 1), end_date: datetime = datetime(2022, 1, 2),
                 tariff: pd.DataFrame = TARIFF,
                 T: int = 1440,
                 *args, **kwargs):

        super().__init__(T=T, grid_node=grid_node, start_date=start_date, end_date=end_date)

        # -> initialize profile generator
        self.profile_generator = StandardLoadProfile(demandP=demandP, london_data=london_data, l_id=l_id,
                                                     resolution=self.T)
        # -> create residents with cars
        self.persons = [Resident(ev_ratio=ev_ratio, charging_limit='required',
                                 start_date=start_date, end_time=end_date, T=self.T)
                        for _ in range(min(2, residents))]

        # -> price limits from survey
        prc_level = round(np.random.normal(loc=MEAN_PRICE, scale=VAR_PRICE), 2)
        self.price_limit = max(1.1477 * (0.805 * prc_level + 17.45) + 1.51, 5)

        tariff.index = pd.date_range(start=datetime(start_date.year, 1, 1), freq='h', periods=len(tariff))
        self.tariff = tariff.resample(RESOLUTION[self.T]).ffill()
        self.grid_fee = pd.Series(index=self.time_range, data=2.6 * np.ones(len(self.time_range)))

        self.pv_system = pv_system
        if self.pv_system:
            self.pv_system = PVSystem(module_parameters=pv_system)
            self.pv_usage = pd.Series(data=np.zeros(self.T), index=self.time_range[:self.T])

        self.model = ConcreteModel()
        self.solver_type = 'glpk'
        self.solver = SolverFactory(self.solver_type)

        self._charging = self.time_range[0] - td(minutes=1)

        cars_ = [person.car for person in self.persons if person.car.type == 'ev']
        if len(cars_) > 0:
            self.total_capacity = sum([c.capacity for c in cars_])
            self.total_benefit = self.total_capacity * self.price_limit
            self.charging = {c: pd.Series(dtype=float) for c in cars_}
        else:
            self.total_capacity = 0
            self.total_benefit = 0

        self.request = pd.Series(dtype=float)

    def set_fixed_demand(self):
        # -> return time series (1/4 h) [kW]
        demand = np.asarray([self.profile_generator.run_model(date) for date in self.date_range]).flatten()
        self._demand = pd.Series(index=self.time_range, data=demand)

    def get_demand(self):
        return self._demand, self._residual_demand

    def get_generation(self):
        return self._generation, self._residual_generation

    def set_photovoltaic_generation(self):
        generation = np.zeros(96 * len(self.date_range))
        if self.pv_system:
            # -> irradiance unit [W/m²]
            irradiance = self.pv_system.get_irradiance(solar_zenith=self.weather['zenith'],
                                                       solar_azimuth=self.weather['azimuth'],
                                                       dni=self.weather['dni'],
                                                       ghi=self.weather['ghi'],
                                                       dhi=self.weather['dhi'])
            # -> get generation in [kW/m^2] * [m^2]
            generation = (irradiance['poa_global'] / 1e3) * self.pv_system.arrays[0].module_parameters['pdc0']
            generation = generation.values
        if self.T == 1440:
            self._generation = pd.Series(index=self.time_range, data=np.repeat(generation, 15))
        elif self.T == 96:
            self._generation = pd.Series(index=self.time_range, data=generation)
        elif self.T == 24:
            generation = np.asarray([np.mean(generation[i:i + 3]) for i in range(0, 96, 4)], np.float).flatten()
            self._generation = pd.Series(index=self.time_range, data=generation)

    def set_residual(self):
        self._residual_demand = self._demand - self._generation
        self._residual_demand[self._residual_demand < 0] = 0
        self._residual_generation = self._generation - self._demand
        self._residual_generation[self._residual_generation < 0] = 0

    def _get_segments(self, steps: int = 20) -> dict:

        x_data = list(np.linspace(0, self.total_capacity, steps))
        y_data = [*map(lambda x: self.total_benefit * (1 - exp(-x / (self.total_capacity / 5))), x_data)]

        logger.debug(f'build piecewise linear function with total-capacity: {self.total_capacity}, '
                     f'total-benefit: {round(self.total_benefit, 1)} divided into {steps} steps')

        segments = dict(low=[], up=[], coeff=[], low_=[])
        for i in range(0, len(y_data) - 1):
            segments['low'].append(x_data[i])
            segments['up'].append(x_data[i + 1])
            dy = (y_data[i + 1] - y_data[i])
            dx = (x_data[i + 1] - x_data[i])
            segments['coeff'].append(dy / dx)
            segments['low_'].append(y_data[i])

        return segments

    def _optimize_photovoltaic_usage(self, d_time: datetime, strategy: str = 'required'):
        logger.debug('optimize charging with photovoltaic generation')

        lin_steps = 10
        # -> get cars and indexer
        cars_ = [person.car for person in self.persons if person.car.type == 'ev']
        cars = range(len(cars_))
        # -> get residual generation and determine possible opt. time steps
        generation = self._residual_generation.loc[d_time:].values
        steps = range(min(self.T, len(generation)))
        # -> get prices
        prices = self.tariff.loc[d_time:].values.flatten()[steps] + self.grid_fee[d_time:].values[steps]
        # -> clear model
        self.model.clear()
        # -> declare variables
        self.model.power = Var(cars, steps, within=Reals, bounds=(0, None))
        self.model.grid = Var(steps, within=Reals, bounds=(0, None))
        self.model.pv = Var(steps, within=Reals, bounds=(0, None))
        self.model.capacity = Var(within=Reals, bounds=(0, self.total_capacity))
        self.model.benefit = Var(within=Reals, bounds=(0, self.total_benefit))

        if self.solver_type == 'glpk':
            segments = self._get_segments(steps=lin_steps)
            s = len(segments['low'])
            self.model.z = Var(range(s), within=Binary)
            self.model.q = Var(range(s), within=Reals)

            # -> segment selection
            self.model.choose_segment = Constraint(expr=quicksum(self.model.z[k] for k in range(s)) == 1)

            self.model.choose_segment_low = ConstraintList()
            self.model.choose_segment_up = ConstraintList()
            for k in range(s):
                self.model.choose_segment_low.add(expr=self.model.q[k] >= segments['low'][k] * self.model.z[k])
                self.model.choose_segment_up.add(expr=self.model.q[k] <= segments['up'][k] * self.model.z[k])

            self.model.benefit_fct = Constraint(expr=quicksum(segments['low_'][k] * self.model.z[k]
                                                              + segments['coeff'][k]
                                                              * (self.model.q[k] - segments['low'][k] * self.model.z[k])
                                                              for k in range(s)) == self.model.benefit)
            self.model.capacity_ct = Constraint(expr=quicksum(self.model.q[k] for k in range(s)) == self.model.capacity)

        elif self.solver_type == 'gurobi':
            logger.debug(f'use gurobi solver to linearize function with total-capacity {self.total_capacity}, '
                         f'total-benefit {round(self.total_benefit, 1)} in {lin_steps} steps')
            x_data = list(np.linspace(0, self.total_capacity, lin_steps))
            y_data = [*map(lambda x: self.total_benefit * (1 - exp(-x / (self.total_capacity / 5))), x_data)]
            self.model.n_soc = Piecewise(self.model.benefit, self.model.capacity,
                                         pw_pts=x_data, f_rule=y_data, pw_constr_type='EQ', pw_repn='SOS2')

        # -> limit maximal charging power
        self.model.power_limit = ConstraintList()
        for c, car in zip(cars, cars_):
            usage = car.usage.loc[d_time:].resample(RESOLUTION[self.T]).last().values
            for t in steps:
                self.model.power_limit.add(self.model.power[c, t] <= car.maximal_charging_power * (1 - usage[t]))

        # -> set range for soc
        self.model.soc_limit = ConstraintList()
        for c, car in zip(cars, cars_):
            limit = car.get_limit(d_time, strategy)
            min_charging = max(car.capacity * limit - car.capacity * car.soc, 0)
            max_charging = car.capacity - car.capacity * car.soc
            self.model.soc_limit.add(quicksum(self.dt * self.model.power[c, t] for t in steps) >= min_charging)
            self.model.soc_limit.add(quicksum(self.dt * self.model.power[c, t] for t in steps) <= max_charging)

        self.model.total_capacity = Constraint(expr=self.model.capacity == quicksum(self.model.power[c, t]
                                                                                    for c in cars for t in steps) * self.dt
                                                    + quicksum(car.capacity * car.soc for car in cars_))
        # -> balance charging, pv and grid consumption
        self.model.balance = ConstraintList()
        for t in steps:
            self.model.balance.add(quicksum(self.model.power[car, t] for car in cars) == self.model.grid[t]
                                   + self.model.pv[t])
        # -> pv range
        self.model.pv_limit = ConstraintList()
        for t in steps:
            self.model.pv_limit.add(self.model.pv[t] <= generation[t])

        self.model.obj = Objective(expr=self.model.benefit - quicksum(prices[t] * self.model.grid[t] * self.dt
                                                                      for t in steps), sense=maximize)
        self.solver.solve(self.model)

        self.request = pd.Series(data=np.asarray([self.model.grid[t].value for t in steps]),
                                 index=pd.date_range(start=d_time, periods=steps[-1] + 1, freq=RESOLUTION[self.T]))

        self.charging = {c: pd.Series(data=np.asarray([self.model.power[c, t].value for t in steps]),
                                      index=pd.date_range(start=d_time, periods=steps[-1] + 1, freq=RESOLUTION[self.T]))
                         for c in cars}

        self.pv_usage = pd.Series(data=np.asarray([self.model.pv[t].value for t in steps]),
                                  index=pd.date_range(start=d_time, periods=steps[-1] + 1, freq=RESOLUTION[self.T]))

        if self.request.sum() < 1e-6 and self.pv_usage.sum() > 0:
            self._charging = self.pv_usage.index[-1]
            for car in cars:
                cars_[car].charging.loc[self.charging[car].index] += self.charging[car]
                self.power.loc[self.charging[car].index] += self.charging[car]

    def _plan_without_photovoltaic(self, d_time: datetime, strategy: str):
        cars = [person.car for person in self.persons if person.car.type == 'ev']

        remaining_steps = len(self.time_range[self.time_range >= d_time])
        self.request = pd.Series(data=np.zeros(remaining_steps),
                                 index=pd.date_range(start=d_time, freq=RESOLUTION[self.T], periods=remaining_steps))
        for car, c in zip(cars, range(len(cars))):
            self.charging[c] = pd.Series(data=np.zeros(remaining_steps),
                                         index=pd.date_range(start=d_time, freq=RESOLUTION[self.T],
                                                             periods=remaining_steps))
            # -> plan charge if the soc < limit and the car is not already charging and the car is at home
            if car.soc < car.get_limit(d_time, strategy) and car.usage[d_time] == 0:
                logger.debug(f'plan charging without photovoltaic for car: {c}')
                # -> get first time stamp of next charging block
                chargeable = car.usage.loc[(car.usage == 0) & (car.usage.index >= d_time)]
                if chargeable.empty:
                    t1 = self.time_range[-1]
                else:
                    t1 = chargeable.index[0]
                # -> get first time stamp of next using block
                car_in_use = car.usage.loc[(car.usage == 1) & (car.usage.index >= d_time)]
                if car_in_use.empty:
                    t2 = self.time_range[-1]
                else:
                    t2 = car_in_use.index[0]
                # -> if d_time in charging block --> plan charging
                if t2 > t1:
                    total_energy = car.capacity - car.capacity * car.soc
                    limit_by_capacity = round(total_energy / car.maximal_charging_power * 1/self.dt)
                    limit_by_slot = len(self.time_range[(self.time_range >= t1) & (self.time_range <= t2)])
                    duration = int(min(limit_by_slot, limit_by_capacity))
                    self.charging[c] = pd.Series(data=car.maximal_charging_power * np.ones(duration),
                                                 index=pd.date_range(start=d_time, freq=RESOLUTION[self.T],
                                                                     periods=duration))
                    # -> add planed charging to power
                    self.request.loc[self.charging[c].index] += self.charging[c].values

                self.request = self.request.loc[self.request > 0]

    def get_request(self, d_time: datetime, strategy: str = 'PV'):
        if self.total_capacity > 0 and d_time > self._charging:
            if strategy == 'PV':
                self._optimize_photovoltaic_usage(d_time=d_time)
            elif strategy == 'simple':
                self._plan_without_photovoltaic(d_time=d_time, strategy='required')
            else:
                logger.error(f'invalid strategy {strategy}')
                raise Exception(f'invalid strategy {strategy}')

        return self.request

    def commit(self, price: pd.Series):
        # price_index = [t.replace(year=2015) for t in price.index]
        total_price = self.tariff.loc[price.index].values.flatten() + price.values
        mean_price = sum(self.request.values * total_price) / sum(self.request.values)
        if mean_price < self.price_limit:
            logger.debug(f'-> commit charging {mean_price}')
            cars_ = [person.car for person in self.persons if person.car.type == 'ev']
            for car in range(len(cars_)):
                cars_[car].charging.loc[self.charging[car].index] += self.charging[car]
                self.power.loc[self.charging[car].index] += self.charging[car]
            self._charging = price.index.max()
            # self.power.loc[self.request.index] += self.request.values
            # self.grid_mobility
            self.request = pd.Series(data=[0], index=[price.index[0]])
            return True
        else:
            logger.debug(f'-> reject charging {mean_price}')
            self.grid_fee[price.index] = price.values
            return False

    def simulate(self, d_time):
        for person in [p for p in self.persons if p.car.type == 'ev']:
            person.car.charge(d_time)  # -> do charging
            person.car.drive(d_time)  # -> do driving


if __name__ == "__main__":
    # -> testing class residential
    from agents.utils import WeatherGenerator
    from datetime import timedelta as td
    from copy import deepcopy

    # -> testing horizon
    start_date = pd.to_datetime('2022-08-01')
    end_date = pd.to_datetime('2022-08-10')
    # -> default pv system
    pv_system = dict(pdc0=5, surface_tilt=35, surface_azimuth=180)
    house_opt = HouseholdModel(residents=2, demandP=5000, pv_system=pv_system, start_date=start_date, end_date=end_date,
                               ev_ratio=1, T=96)
    # # -> get weather data
    weather_generator = WeatherGenerator()
    weather = pd.concat([weather_generator.get_weather(area='DEA26', date=date)
                         for date in pd.date_range(start=start_date, end=end_date + td(days=1),
                                                   freq='d')])
    weather = weather.resample('15min').ffill()
    weather = weather.loc[weather.index.isin(house_opt.time_range)]
    # -> set parameter
    house_opt.set_parameter(weather=weather)
    house_opt.set_fixed_demand()
    house_opt.set_photovoltaic_generation()
    house_opt.set_residual()
    # opt_power = house_opt.get_request(house_opt.time_range[0], strategy='PV')
    time_range = house_opt.time_range
    # -> clone house_1
    house_simple = deepcopy(house_opt)
    # offset = [100] * 10

    for t in time_range:
        # print(t)
        simple_power = house_simple.get_request(t, strategy='simple')
        if sum(simple_power) > 0:
            house_simple.commit(pd.Series(data=np.ones(len(simple_power)), index=simple_power.index))
        house_simple.simulate(d_time=t)
    #
        opt_power = house_opt.get_request(t, strategy='PV')
        if sum(opt_power) > 0.01:
            house_opt.commit(pd.Series(data=np.zeros(len(opt_power)), index=opt_power.index))
        house_opt.simulate(d_time=t)
        break
    # plt.plot(house_opt.power)
    # plt.plot(house_opt.persons[0].car.charging + house_opt.persons[1].car.charging)
    # plt.show()
    # for house, strategy in zip([house_opt, house_simple], ['opt', 'simple']):
    #     generation, residual_generation = house.get_generation()
    #     demand, residual_demand = house.get_demand()
    #     charged = house.power
    #     cars = [person.car.monitor for person in house.persons if person.car.type == 'ev']
    #     cars = pd.concat(cars, axis=1)
    #     prices = house.tariff.loc[house.time_range]
    #     result = pd.concat([prices, demand, residual_generation, charged, cars], axis=1)
    #     result.columns = ['prices', 'demand', 'generation', 'charged',
    #                       'distance_1', 'odometer_1', 'soc_1', 'work_1', 'errand_1', 'hobby_1',
    #                       'distance_2', 'odometer_2', 'soc_2', 'work_2', 'errand_2', 'hobby_2']
    #     result.to_excel(f'house_{strategy}.xlsx')
        # opt_power = house_1.get_request(t, strategy='PV')
        # if sum(opt_power) > 0:
        #     print('get order house_1')
        #     # house_1.commit(pd.Series(data=np.zeros(len(opt_power)), index=opt_power.index))
        #     plt.plot(opt_power)
        #     plt.show()



    # # assert cap <= house.persons[0].car.capacity/2
    # # print(cap, house.persons[0].car.capacity)
    # plt.plot(power.values)
    # plt.show()
    #
    # # car_2 = np.asarray([model.power[1, t].value for t in range(1440)])
    # # for t in pd.date_range(start=start_date, periods=1440, freq='min'):
    # #    rq = house.get_request(t)
    # #    house.simulate(t)
    # #    break
