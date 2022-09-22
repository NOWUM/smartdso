from math import exp
import numpy as np
import pandas as pd
import os
from datetime import datetime, timedelta as td
import logging
from pvlib.pvsystem import PVSystem
import string
import itertools
import uuid
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


SEED = int(os.getenv('RANDOM_SEED', 2022))
random = np.random.default_rng(SEED)


letters = list(string.ascii_uppercase)
letters = [f'{a}{b}{c}' for a, b, c in itertools.product(letters, letters, letters)]
numbers = [f"{number:04d}" for number in range(1_000)]
KEYS = [f'{a}{b}' for a, b in itertools.product(letters, numbers)]


# -> price data from survey
# MEAN_PRICE = 28.01
# VAR_PRICE = 7.9
# -> steps and corresponding time resolution strings in pandas
RESOLUTION = {1440: 'min', 96: '15min', 24: 'h'}
# -> timescaledb connection to store the simulation results
DATABASE_URI = os.getenv('DATABASE_URI', 'postgresql://opendata:opendata@10.13.10.41:5432/smartdso')
# -> default prices EPEX-SPOT 2015
TARIFF = pd.read_csv(r'./participants/data/2022_prices.csv', index_col=0)
TARIFF = TARIFF / 10  # -> [€/MWh] in [ct/kWh]
# TARIFF.index = pd.to_datetime(TARIFF.index, format="%d.%m.%Y %H:%M")
TARIFF.index = pd.to_datetime(TARIFF.index, format="%Y-%m-%d %H:%M:%S")
CHARGES = {'others': 2.9, 'taxes': 8.0}
for values in CHARGES.values():
    TARIFF += values


class HouseholdModel(BasicParticipant):

    def __init__(self, residents: int,
                 demandP: float, london_data: bool = True, l_id: str = 'MAC002957',
                 ev_ratio: float = 0.5,
                 pv_systems: list = None,
                 grid_node: str = None,
                 start_date: datetime = datetime(2022, 1, 1), end_date: datetime = datetime(2022, 1, 2),
                 tariff: pd.DataFrame = TARIFF,
                 T: int = 1440,
                 database_uri: str = DATABASE_URI,
                 consumer_id: str = 'nowum',
                 price_sensitivity: float = 1.3,
                 strategy: str = 'MaxPvCap',
                 scenario: str = 'Flat',
                 random: np.random.default_rng = random,
                 *args, **kwargs):

        super().__init__(T=T, grid_node=grid_node, start_date=start_date, end_date=end_date,
                         database_uri=database_uri, consumer_type='household', strategy=strategy,
                         random=random)

        # -> initialize profile generator
        self._profile_generator = StandardLoadProfile(demandP=demandP, london_data=london_data,
                                                      l_id=l_id, resolution=self.T)
        # -> create residents with cars
        self.persons = [Resident(ev_ratio=ev_ratio, charging_limit='required',
                                 start_date=start_date, end_time=end_date, T=self.T)
                        for _ in range(min(2, residents))]

        # -> price limits from survey
        if 'MaxPv' in strategy:
            self.price_limit = 45
            self._slope = price_sensitivity
        elif 'PlugInInf' in strategy:
            self.price_limit = np.inf
            self._slope = 0
        else:
            self.price_limit = 45
            self._slope = 0

        self._pv_systems = [PVSystem(module_parameters=system) for system in pv_systems]

        self.cars = dict()

        for person in self.persons:
            if person.car.type == 'ev':
                key = list(random.choice(KEYS))
                random.shuffle(key)
                key = ''.join(key)
                self.cars[key] = person.car

        if len(self.cars) > 0:
            self._total_capacity = sum([c.capacity for c in self.cars.values()])
            self._total_benefit = self._total_capacity * self.price_limit
            self._car_power = {c: pd.Series(dtype=float) for c in self.cars.keys()}
        else:
            self._total_capacity = 0
            self._total_benefit = 0

        self._max_requests = 5
        self._benefit_value = 0

        self._simple_commit = None
        self.finished = False

        self._model = ConcreteModel()
        self._solver_type = 'glpk'
        self._solver = SolverFactory(self._solver_type)

        tariff.index = pd.date_range(start=datetime(start_date.year, 1, 1), freq='h', periods=len(tariff))
        tariff = tariff.resample(RESOLUTION[self.T]).ffill().loc[self.time_range]
        self._data.loc[tariff.index, 'tariff'] = tariff.values.flatten()
        if 'Flat' in scenario:
            # -> use median
            median_value = np.sort(tariff.values.flatten())[int(len(tariff)/2)]
            self._data.loc[tariff.index, 'tariff'] = median_value
        self._data.loc[self.time_range, 'grid_fee'] = 2.6 * np.ones(self._steps)
        pv_capacity = sum([s['pdc0'] for s in pv_systems])
        self._data.loc[self.time_range, 'pv_capacity'] = pv_capacity * np.ones(self._steps)
        self._data.loc[self.time_range, 'consumer_id'] = [consumer_id] * self._steps
        self._data.loc[self.time_range, 'car_capacity'] = self._total_capacity * np.ones(self._steps)

    def _get_segments(self, steps: int = 20) -> dict:

        x_data = list(np.linspace(0, self._total_capacity, steps))
        y_data = [*map(lambda x: self._total_benefit * (1 - exp(-x / (self._total_capacity / self._slope))), x_data)]

        logger.debug(f'build piecewise linear function with total-capacity: {self._total_capacity}, '
                     f'total-benefit: {round(self._total_benefit, 1)} divided into {steps} steps')

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
        # -> get residual generation and determine possible opt. time steps
        generation = self._data.loc[d_time:, 'residual_generation'].values
        steps = range(min(self.T, len(generation)))
        demand = {key: car.get_data('demand').loc[d_time:].values[steps] for key, car in self.cars.items()}
        # -> get prices
        tariff = self._data.loc[d_time:, 'tariff'].values.flatten()[steps]
        grid_fee = self._data.loc[d_time:, 'grid_fee'].values.flatten()[steps]
        prices = tariff + grid_fee
        if self._max_requests == 0:
            prices += 1e6

        # -> clear model
        self._model.clear()
        # -> declare variables
        self._model.power = Var(self.cars.keys(), steps, within=Reals, bounds=(0, None))
        self._model.grid = Var(steps, within=Reals, bounds=(0, None))
        self._model.pv = Var(steps, within=Reals, bounds=(0, None))
        self._model.capacity = Var(within=Reals, bounds=(0, self._total_capacity))

        self._model.benefit = Var(within=Reals, bounds=(0, self._total_benefit))
        if 'Soc' in self._used_strategy:
            if self._solver_type == 'glpk':
                segments = self._get_segments(steps=lin_steps)
                s = len(segments['low'])
                self._model.z = Var(range(s), within=Binary)
                self._model.q = Var(range(s), within=Reals)

                # -> segment selection
                self._model.choose_segment = Constraint(expr=quicksum(self._model.z[k] for k in range(s)) == 1)

                self._model.choose_segment_low = ConstraintList()
                self._model.choose_segment_up = ConstraintList()
                for k in range(s):
                    self._model.choose_segment_low.add(expr=self._model.q[k] >= segments['low'][k] * self._model.z[k])
                    self._model.choose_segment_up.add(expr=self._model.q[k] <= segments['up'][k] * self._model.z[k])

                self._model.benefit_fct = Constraint(expr=quicksum(segments['low_'][k] * self._model.z[k]
                                                                   + segments['coeff'][k]
                                                                   * (self._model.q[k] - segments['low'][k] * self._model.z[k])
                                                                   for k in range(s)) == self._model.benefit)
                self._model.capacity_ct = Constraint(expr=quicksum(self._model.q[k] for k in range(s)) == self._model.capacity)

            elif self._solver_type == 'gurobi':
                logger.debug(f'use gurobi solver to linearize function with total-capacity {self._total_capacity}, '
                             f'total-benefit {round(self._total_benefit, 1)} in {lin_steps} steps')
                x_data = list(np.linspace(0, self._total_capacity, lin_steps))
                y_data = [*map(lambda x: self._total_benefit * (1 - exp(-x / (self._total_capacity / self._slope))), x_data)]
                self._model.n_soc = Piecewise(self._model.benefit, self._model.capacity,
                                              pw_pts=x_data, f_rule=y_data, pw_constr_type='EQ', pw_repn='SOS2')
        else:
            self._model.benefit_fct = Constraint(expr=self._model.benefit == self.price_limit * self._model.capacity)

        # -> limit maximal charging power
        self._model.power_limit = ConstraintList()
        for key, car in self.cars.items():
            usage = car.get_data('usage').loc[d_time:].values
            for t in steps:
                if car.soc < car.get_limit(d_time, strategy):
                    max_power = car.maximal_charging_power * (1 - usage[t])
                else:
                    max_power = 0
                self._model.power_limit.add(self._model.power[key, t] <= max_power)

        # -> set range for soc
        self._model.soc_limit = ConstraintList()
        for key, car in self.cars.items():
            limit = car.get_limit(d_time, strategy)
            min_charging = max(car.capacity * (limit - car.soc), 0)
            max_charging = car.capacity * (1 - car.soc)
            self._model.soc_limit.add(quicksum(self.dt * self._model.power[key, t] for t in steps) >= min_charging)
            self._model.soc_limit.add(quicksum(self.dt * self._model.power[key, t] for t in steps) <= max_charging)

        self._model.total_capacity = Constraint(expr=self._model.capacity == quicksum(self._model.power[key, t]
                                                                                      - demand[key][t]
                                                                                      for key in self.cars.keys()
                                                                                      for t in steps) * self.dt
                                                    + quicksum(car.capacity * car.soc for car in self.cars.values()))
        # -> balance charging, pv and grid consumption
        self._model.balance = ConstraintList()
        for t in steps:
            self._model.balance.add(quicksum(self._model.power[key, t] for key in self.cars.keys())
                                    == self._model.grid[t] + self._model.pv[t])

        # -> pv range
        self._model.pv_limit = ConstraintList()
        for t in steps:
            self._model.pv_limit.add(self._model.pv[t] <= generation[t])

        self._model.obj = Objective(expr=self._model.benefit - quicksum(prices[t] * self._model.grid[t] * self.dt
                                                                        for t in steps), sense=maximize)

        time_range = pd.date_range(start=d_time, periods=len(steps), freq=RESOLUTION[self.T])
        self._request = pd.Series(data=np.zeros(len(steps)), index=time_range)
        self._car_power = {key: pd.Series(data=np.zeros(len(steps)), index=time_range) for key in self.cars.keys()}

        try:
            self._solver.solve(self._model)
            self._benefit_value = value(self._model.benefit)
            self._request.loc[time_range] = np.round(np.asarray([self._model.grid[t].value for t in steps]), 2)
            for key in self.cars.keys():
                self._car_power[key].loc[time_range] = [self._model.power[key, t].value for t in steps]
            if self._initial_plan:
                self._data.loc[time_range, 'planned_grid_consumption'] = self._request.loc[time_range].copy()
                self._data.loc[time_range, 'planned_pv_consumption'] = [self._model.pv[t].value for t in steps]
                for key, car in self.cars.items():
                    car.set_planned_charging(self._car_power[key])
                self._initial_plan = False

        except Exception as e:
            logger.warning(f' -> model infeasible {repr(e)}')
            print(self._request.sum())

        if self._request.sum() == 0:
            self._commit = time_range[-1]
            for key, car in self.cars.items():
                car.set_final_charging(self._car_power[key])
            self._data.loc[time_range, 'final_pv_consumption'] = [self._model.pv[t].value for t in steps]

            self._max_requests = 5
            self._finished = True
            self._initial_plan = True

    def _plan_without_photovoltaic(self, d_time: datetime, strategy: str = 'required'):
        self._simple_commit = d_time
        d_time += td(minutes=15 * int(sum(self.random.integers(low=1, high=3, size=(5-self._max_requests)))))
        remaining_steps = min(len(self.time_range[self.time_range >= d_time]), self.T)
        generation = self._data.loc[d_time:d_time + td(hours=(remaining_steps-1)*self.dt), 'residual_generation']
        self._request = pd.Series(data=np.zeros(remaining_steps),
                                  index=pd.date_range(start=d_time, freq=RESOLUTION[self.T], periods=remaining_steps))

        t_next_request = d_time

        for key, car in self.cars.items():
            self._car_power[key] = pd.Series(data=np.zeros(remaining_steps),
                                             index=pd.date_range(start=d_time, freq=RESOLUTION[self.T],
                                                                 periods=remaining_steps))
            usage = car.get_data('usage').loc[d_time:]
            if car.soc < car.get_limit(d_time, strategy) and usage.at[d_time] == 0:
                chargeable = usage.loc[usage == 0]
                # -> get first time stamp of next charging block
                if chargeable.empty:
                    t1 = self.time_range[-1]
                else:
                    t1 = chargeable.index[0]
                # -> get first time stamp of next using block
                car_in_use = usage.loc[usage == 1]
                if car_in_use.empty:
                    t2 = self.time_range[-1]
                else:
                    t2 = car_in_use.index[0]

                if t2 > t1:
                    limit_by_capacity = (car.capacity * (1-car.soc)) / car.maximal_charging_power / self.dt
                    limit_by_slot = len(self.time_range[(self.time_range >= t1) & (self.time_range < t2)])
                    duration = int(min(limit_by_slot, limit_by_capacity))
                    self._car_power[key] = pd.Series(data=car.maximal_charging_power * np.ones(duration),
                                                     index=pd.date_range(start=d_time, freq=RESOLUTION[self.T],
                                                                         periods=duration))

                    self._request.loc[self._car_power[key].index] += self._car_power[key].values
                    if t_next_request == d_time:
                        t_next_request = t2
                    else:
                        t_next_request = min(t_next_request, t2)

        for key, car in self.cars.items():
            car.set_planned_charging(self._car_power[key])

        if self._request.sum() > 0:
            pv_usage = pd.Series(data=np.zeros(remaining_steps),
                                 index=pd.date_range(start=d_time, freq=RESOLUTION[self.T], periods=remaining_steps))

            generation.loc[generation.values > max(self._request.values)] = max(self._request.values)
            pv_usage.loc[self._request > 0] = generation.loc[self._request > 0].values

            self._request.loc[self._request > 0] -= generation.loc[self._request > 0]
            self._simple_commit = t_next_request
            if self._initial_plan:
                self._initial_plan = False
                self._data.loc[self._request.index, 'planned_grid_consumption'] = self._request.values.copy()
                self._data.loc[pv_usage.index, 'planned_pv_consumption'] = pv_usage.copy()

            capacity = sum([car.soc * car.capacity for car in self.cars.values()]) + self._request.values.sum() * self.dt
            self._benefit_value = self.price_limit * capacity

        else:
            self._commit = t_next_request
            self._simple_commit = t_next_request

    def get_request(self, d_time: datetime, strategy: str = 'MaxPvCap'):
        self._used_strategy = strategy
        if self._total_capacity > 0 and d_time > self._commit:
            if 'MaxPv' in strategy:
                self._optimize_photovoltaic_usage(d_time=d_time)
            elif 'PlugIn' in strategy:
                self._plan_without_photovoltaic(d_time=d_time)
            else:
                logger.error(f'invalid strategy {strategy}')
                raise Exception(f'invalid strategy {strategy}')

        elif self._total_capacity == 0:
            self._finished = True
            self._initial_plan = True

        return self._request

    def commit(self, price: pd.Series):
        tariff = self._data.loc[price.index, 'tariff'].values.flatten()
        grid_fee = price.values.flatten()
        total_price = sum((tariff + grid_fee) * self._request.values) * self.dt
        if self._benefit_value > total_price or self._max_requests == 0:
            for key, car in self.cars.items():
                car.set_final_charging(self._car_power[key])
            self._commit = self._simple_commit or price.index.max()
            self._data.loc[price.index, 'final_grid_consumption'] = self._request.loc[price.index].copy()
            self._data.loc[:, 'final_pv_consumption'] = self._data.loc[:, 'planned_pv_consumption'].copy()
            self._request = pd.Series(data=np.zeros(len(price)), index=price.index)
            self._data.loc[price.index, 'grid_fee'] = price.values
            self._max_requests = 5
            if 'MaxPv' in self._used_strategy:
                self._finished = True
            self._initial_plan = True
            return True
        else:
            self._data.loc[price.index, 'grid_fee'] = price.values
            self._request = pd.Series(data=np.zeros(len(price)), index=price.index)
            self._max_requests -= 1
            return False


if __name__ == "__main__":
    # -> testing class residential
    from agents.utils import WeatherGenerator
    from datetime import timedelta as td

    # -> testing horizon
    start_date = pd.to_datetime('2022-01-01')
    end_date = pd.to_datetime('2022-01-10')
    # -> default pv system
    pv_system = dict(pdc0=5, surface_tilt=35, surface_azimuth=180)
    house_opt = HouseholdModel(residents=1, demandP=5000, pv_systems=[pv_system],
                               start_date=start_date, end_date=end_date, ev_ratio=1, T=96,
                               strategy='MaxPvCap')
    # # -> get weather data
    weather_generator = WeatherGenerator()
    weather = pd.concat([weather_generator.get_weather(date=date)
                         for date in pd.date_range(start=start_date, end=end_date + td(days=1),
                                                   freq='d')])
    weather = weather.resample('15min').ffill()
    weather = weather.loc[weather.index.isin(house_opt.time_range)]
    # -> set parameter
    house_opt.set_parameter(weather=weather)
    house_opt.initial_time_series()
    #opt_power = house_opt.get_request(house_opt.time_range[0], strategy='PV')
    time_range = house_opt.time_range
    t = time_range[0]
    #r = house_opt.get_request(d_time=t, strategy='simple')
    for t in time_range:
        x = house_opt.get_request(d_time=t, strategy='MaxPvCap')
        # print(t, x)
        if house_opt._request.sum() > 0:
            print(f'send request at {t}')
            house_opt.commit(pd.Series(data=0.5 * np.ones(len(house_opt._request)), index=house_opt._request.index))
           #commit = False
           # while not commit:
           #    commit = house_opt.commit(pd.Series(data=1500*np.ones(len(house_opt._request)), index=house_opt._request.index))
        house_opt.simulate(t)
    result = house_opt.get_result()
    # -> clone house_1
    # house_simple = deepcopy(house_opt)
    # offset = [100] * 10

    # for t in time_range:
    #     print(t, house_opt.has_commit(), house_opt._commit)
    #     opt_power = house_opt.get_request(t, strategy='PV')
    #     if sum(opt_power) > 0:
    #         house_opt.commit(pd.Series(data=np.zeros(len(opt_power)), index=opt_power.index))
    #     house_opt.simulate(d_time=t)
    #
    # plt.plot(house_opt._data['final_pv_consumption'])
    # plt.show()
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
