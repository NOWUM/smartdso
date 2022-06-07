from math import exp
import numpy as np
import pandas as pd
from datetime import datetime
from pvlib.pvsystem import PVSystem
from pyomo.environ import Constraint, Var, Objective, SolverFactory, ConcreteModel, \
    Reals, Binary, minimize, maximize, value, quicksum, ConstraintList, Piecewise

# example to Piecewise: http://yetanothermathprogrammingconsultant.blogspot.com/2019/02/piecewise-linear-functions
# -and.html

from participants.basic import BasicParticipant
from participants.utils import Resident
from demLib.electric_profile import StandardLoadProfile

# -> price data from survey
MEAN_PRICE = 28.01
VAR_PRICE = 7.9

# -> default prices EPEX SPOT 2015
TARIFF = pd.read_csv(r'./participants/data/default_prices.csv', index_col=0)
TARIFF = TARIFF.values.flatten().repeat(60)
TARIFF = pd.Series(data=TARIFF, index=pd.date_range(start='2015-01-01', periods=len(TARIFF), freq='min'))


class HouseholdModel(BasicParticipant):

    def __init__(self, residents: int,
                 demandP: float, london_data: bool = False, l_id: str = None,
                 ev_ratio: float = 0.5, minimum_soc: int = -1,
                 pv_system: dict = None,
                 grid_node: str = None,
                 start_date: datetime = datetime(2022, 1, 1), end_date: datetime = datetime(2022, 1, 2),
                 tariff: pd.DataFrame = TARIFF,
                 T: int = 1440,
                 *args, **kwargs):

        super().__init__(T=T, grid_node=grid_node, start_date=start_date, end_date=end_date)

        # -> initialize profile generator
        self.profile_generator = StandardLoadProfile(demandP=demandP, london_data=london_data, l_id=l_id)
        # -> create residents with cars
        self.persons = [Resident(ev_ratio=ev_ratio, minimum_soc=minimum_soc, base_price=29,
                                 start_date=start_date, end_time=end_date)
                        for _ in range(min(2, residents))]

        # -> price limits from survey
        prc_level = round(np.random.normal(loc=MEAN_PRICE, scale=VAR_PRICE), 2)
        self.price_limit = max(1.1477 * (0.805 * prc_level + 17.45) + 1.51, 5)

        self.tariff = tariff
        self.pv_system = pv_system
        if self.pv_system:
            self.pv_system = PVSystem(module_parameters=pv_system)

        self.model = ConcreteModel()
        self.solver = SolverFactory('gurobi')

        self.delay = 0
        self.waiting_time = 0

        self.power = None
        self.charging = None
        self._charging = 0
        self.grid_fee = pd.Series(index=self.time_range, data=2.6 * np.ones(len(self.time_range)))

    def set_fixed_demand(self):
        # -> return time series (1/4 h) [kW]
        demand = np.asarray([self.profile_generator.run_model(date) for date in self.date_range]).flatten()
        self._demand = pd.Series(index=self.time_range, data=np.repeat(demand, 15))

    def get_demand(self):
        return self._demand, self._residual_demand

    def get_generation(self):
        return self._generation, self._residual_generation

    def set_photovoltaic_generation(self):
        generation = np.zeros(96*len(self.date_range))
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
        # resample to minute resolution
        self._generation = pd.Series(index=self.time_range, data=np.repeat(generation, 15))

    def set_residual(self):
        self._residual_demand = self._demand - self._generation
        self._residual_demand[self._residual_demand < 0] = 0
        self._residual_generation = self._generation - self._demand
        self._residual_generation[self._residual_generation < 0] = 0

    def get_request(self, d_time: datetime):
        cars_ = [person.car for person in self.persons if person.car.type == 'ev']
        if self._charging == 0 and cars_ and any([car.soc < 95 for car in cars_]):
            cars = range(len(cars_))
            # -> calculate total capacity of all cars and the total benefit
            total_capacity = sum([c.capacity for c in cars_])
            total_benefit = total_capacity * self.price_limit
            # -> get residual generation
            _, g = self.get_generation()
            g = g.loc[d_time:].values
            steps = range(min(self.T, len(g)))
            # -> get prices
            tariff = self.tariff.loc[d_time.replace(year=2015):].values
            grid_fee = self.grid_fee[d_time:].values
            prices = tariff[steps] + grid_fee[steps]

            # -> build model
            self.model.clear()
            self.model.power = Var(cars, steps, within=Reals, bounds=(0, None))
            self.model.grid = Var(steps, within=Reals, bounds=(0, None))
            self.model.pv = Var(steps, within=Reals, bounds=(0, None))
            self.model.capacity = Var(within=Reals, bounds=(0, total_capacity))
            self.model.benefit = Var(within=Reals, bounds=(0, total_benefit))

            # -> linearize benefit function
            x_data = list(np.linspace(0, total_capacity, 20))
            y_data = [*map(lambda x: total_benefit*(1-exp(-x/(total_capacity / 5))), x_data)]
            self.model.n_soc = Piecewise(self.model.benefit, self.model.capacity,
                                         pw_pts=x_data, f_rule=y_data,
                                         pw_constr_type='EQ', pw_repn='SOS2')

            # -> limit maximal charging power
            self.model.power_limit = ConstraintList()
            for c, car in zip(cars, cars_):
                usage = car.usage.loc[d_time:].values
                for t in steps:
                    self.model.power_limit.add(self.model.power[c, t] <= car.maximal_charging_power * (1 - usage[t]))

            # -> set range for soc
            self.model.soc_limit = ConstraintList()
            for c, car in zip(cars, cars_):
                limit = car.get_limit(d_time)
                min_capacity = car.capacity * (limit / 100) - (car.capacity * (car.soc / 100))
                max_capacity = car.capacity - (car.capacity * (car.soc / 100))
                self.model.soc_limit.add(quicksum(1 / 60 * self.model.power[c, t] for t in steps) >= min_capacity)
                self.model.soc_limit.add(quicksum(1 / 60 * self.model.power[c, t] for t in steps) <= max_capacity)

            self.model.total_capacity = Constraint(expr=self.model.capacity == 1 / 60 * quicksum(self.model.power[c, t]

                                                                                                 for c in cars for t in
                                                                                                 steps))
            # -> balance charging, pv and grid consumption
            self.model.balance = ConstraintList()
            for t in steps:
                self.model.balance.add(quicksum(self.model.power[car, t] for car in cars)
                                       == self.model.grid[t] + self.model.pv[t])

            # -> pv range
            self.model.pv_limit = ConstraintList()
            for t in steps:
                self.model.pv_limit.add(self.model.pv[t] <= g[t])

            self.model.obj = Objective(expr=self.model.benefit - quicksum(prices[t] / 60 * self.model.grid[t]
                                                                          for t in steps), sense=maximize)
            self.solver.solve(self.model)

            self.power = pd.Series(data=np.asarray([self.model.grid[t].value for t in steps]),
                                   index=pd.date_range(start=d_time, periods=steps[-1] + 1, freq='min'))

            self.charging = {c: pd.Series(data=np.asarray([self.model.power[c, t].value for t in steps]),
                                   index=pd.date_range(start=d_time, periods=steps[-1] + 1, freq='min'))
                             for c in cars}
        else:
            self.power = pd.Series(data=np.zeros(self.T), index=pd.date_range(start=d_time, periods=self.T, freq='min'))

        return self.power

    def commit(self, price: pd.Series):
        mean_price = sum((self.power.values/60) * price.values)/sum((self.power.values/60))
        if mean_price < self.price_limit:
            self._charging = 1440
            cars_ = [person.car for person in self.persons if person.car.type == 'ev']
            for car in range(len(cars_)):
                cars_[car].charging = self.charging[car]
                cars_[car].charging = True
            return True
        else:
            self.grid_fee[price.index] = price.values
            return False

    def simulate(self, d_time):
        for person in [p for p in self.persons if p.car.type == 'ev']:
            if self._charging == 0:
                person.car.charging = False
            person.car.charge(d_time)  # -> do charging
            person.car.drive(d_time)  # -> do driving

        if self._charging > 0:
            self._charging = -1



if __name__ == "__main__":
    from matplotlib import pyplot as plt
    from agents.utils import WeatherGenerator
    from datetime import timedelta as td

    plotting = True

    start_date = pd.to_datetime('2022-08-01')
    end_date = pd.to_datetime('2022-08-10')

    pv_system = dict(pdc0=5, surface_tilt=35, surface_azimuth=180)
    house = HouseholdModel(residents=5, demandP=5000, pv_system=pv_system, start_date=start_date, end_date=end_date,
                           ev_ratio=1, base_price=10)

    weather_generator = WeatherGenerator()
    weather = pd.concat([weather_generator.get_weather(area='DEA26', date=date)
                         for date in pd.date_range(start=start_date, end=end_date + td(days=1),
                                                   freq='d')])
    weather = weather.resample('15min').ffill()
    weather = weather.loc[weather.index.isin(house.time_range)]
    house.set_parameter(weather=weather)
    house.set_fixed_demand()
    house.set_photovoltaic_generation()
    house.set_residual()

    if plotting:
        demand, res_demand = house.get_demand()
        res_demand.plot()
        plt.show()
        generation, res_generation = house.get_generation()
        res_generation.plot()
        plt.show()

    ts = house.time_range[0]
    _, gen = house.get_generation()
    plt.plot(house.persons[0].car.usage.values[:1440])
    plt.plot(gen[ts:].values[:1440])
    #for offset in range(0, 50, 10):
    house.tariff += 0
    power = house.get_request(ts)
    # assert cap <= house.persons[0].car.capacity/2
    # print(cap, house.persons[0].car.capacity)
    plt.plot(power.values)
    plt.show()



    # car_2 = np.asarray([model.power[1, t].value for t in range(1440)])
    # for t in pd.date_range(start=start_date, periods=1440, freq='min'):
    #    rq = house.get_request(t)
    #    house.simulate(t)
    #    break
