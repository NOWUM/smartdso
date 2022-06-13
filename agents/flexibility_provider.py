import uuid
import pandas as pd
import numpy as np
from datetime import datetime, timedelta as td

from participants.residential import HouseholdModel
from participants.business import BusinessModel
from participants.industry import IndustryModel
from agents.utils import WeatherGenerator

# -> read known consumers and nodes
consumers = pd.read_csv(r'./gridLib/data/grid_allocations.csv', index_col=0)
h0_consumers = consumers.loc[consumers['profile'] == 'H0']
g0_consumers = consumers.loc[consumers['profile'] == 'G0']
rlm_consumers = consumers.loc[consumers['profile'] == 'RLM']

RESOLUTION = {1440: 'min', 96: '15min', 24: 'h'}


class FlexibilityProvider:

    def __init__(self, scenario: str, iteration: int, dynamic_fee: bool,
                 start_date: datetime, end_date: datetime, ev_ratio: float = 0.5,
                 minimum_soc: int = -1, london_data: bool = False, pv_ratio: float = 0.3,
                 T: int = 1440, *args, **kwargs):

        # -> scenario name and iteration number
        self.scenario = scenario
        self.iteration = iteration
        # -> economic settings
        self.dynamic_fee = dynamic_fee
        # -> total clients
        self.clients = {}
        self._rq_id = None
        # -> weather generator
        self.weather_generator = WeatherGenerator()
        # -> time range
        self.time_range = pd.date_range(start=start_date, end=end_date + td(days=1), freq=RESOLUTION[T])[:-1]
        self.indexer = 0
        len_ = len(self.time_range)
        # -> simulation monitoring
        self.capacity = 0
        self.power = 0
        self.empty_counter = np.zeros(len_)
        self.virtual_source = np.zeros(len_)
        self.prices = np.zeros(len_)
        self.charged = pd.Series(index=self.time_range, data=np.zeros(len_))
        self.shifted = np.zeros(len_)
        self.sub_grid = -1*np.ones(len_)
        self.soc = np.zeros(len_)
        self.cost = np.zeros(len_)
        self.pv_capacity = 0

        # -> create household clients
        for _, consumer in h0_consumers.iterrows():

            if np.random.choice(a=[True, False], p=[pv_ratio, 1-pv_ratio]):
                pv_system = dict(pdc0=consumer['photovoltaic_potential'], surface_tilt=35, surface_azimuth=180)
                self.pv_capacity += consumer['photovoltaic_potential']
            else:
                pv_system = None

            client = HouseholdModel(demandP=consumer['jeb'],
                                    residents=int(max(consumer['jeb'] / 1500, 1)),
                                    london_data=london_data,
                                    l_id=consumer['london_data'],
                                    minimum_soc=minimum_soc,
                                    ev_ratio=ev_ratio,
                                    pv_system=pv_system,
                                    start_date=start_date,
                                    end_date=end_date,
                                    T=T,
                                    grid_node=consumer['bus0'])

            for person in [p for p in client.persons if p.car.type == 'ev']:
                self.capacity += person.car.capacity
                self.power += person.car.maximal_charging_power
            self.clients[uuid.uuid1()] = client

        # -> select reference car
        self.reference_car = None
        if ev_ratio > 0:
            while self. reference_car is None:
                key = np.random.choice([key for key in self.clients.keys()])
                for person in [p for p in self.clients[key].persons if p.car.type == 'ev']:
                    self.reference_car = person.car

        # -> create business clients
        for _, consumer in g0_consumers.iterrows():
            client = BusinessModel(T=T, demandP=consumer['jeb'], grid_node=consumer['bus0'],
                                   start_date=start_date, end_date=end_date)
            self.clients[uuid.uuid1()] = client

        # -> create industry clients
        for _, consumer in rlm_consumers.iterrows():
            client = IndustryModel(T=T, demandP=consumer['jeb'], grid_node=consumer['bus0'],
                                   start_date=start_date, end_date=end_date)
            self.clients[uuid.uuid1()] = client

    def initialize_time_series(self) -> (pd.DataFrame, pd.DataFrame):

        def build_dataframe(data, id_):
            dataframe = pd.DataFrame({'power': data.values})
            dataframe.index = self.time_range
            dataframe['id_'] = str(id_)
            dataframe['node_id'] = client.grid_node
            dataframe = dataframe.rename_axis('t')
            return dataframe

        weather = pd.concat([self.weather_generator.get_weather(area='DEA26', date=date)
                            for date in pd.date_range(start=self.time_range[0], end=self.time_range[-1] + td(days=1),
                                                      freq='d')])
        weather = weather.resample('15min').ffill()
        weather = weather.loc[weather.index.isin(self.time_range)]

        demand_, generation_ = [], []
        for id_, client in self.clients.items():
            client.set_parameter(weather=weather.copy())
            client.set_photovoltaic_generation()
            client.set_fixed_demand()
            client.set_residual()
            _, demand = client.get_demand()
            demand_.append(build_dataframe(demand, id_))
            _, generation = client.get_generation()
            generation_.append(build_dataframe(generation, id_))

        return pd.concat(demand_), pd.concat(generation_)

    def get_requests(self, d_time: datetime) -> (pd.Series, str):
        for id_, client in self.clients.items():
            request = client.get_request(d_time)
            if sum(request.values) > 1e-6:
                yield request, client.grid_node, id_

    def simulate(self, d_time: datetime) -> None:
        capacity, empty, pool = 0, 0, 0
        for participant in self.clients.values():
            participant.simulate(d_time)
            for person in [p for p in participant.persons if p.car.type == 'ev']:
                capacity += person.car.soc * person.car.capacity
                empty += int(person.car.empty)
                pool += person.car.virtual_source
        self.soc[self.indexer] = capacity / self.capacity
        # -> add to simulation monitoring
        self.empty_counter[self.indexer] = empty
        self.virtual_source[self.indexer] = pool
        # -> increment time counter
        self.indexer += 1

    def commit(self, price: pd.Series, request: pd.Series, consumer_id: uuid.uuid1) -> bool:
        commit_ = self.clients[consumer_id].commit(price=price)
        if commit_:
            self.charged.loc[request.index] += request.values
        return commit_

    def get_results(self) -> (pd.DataFrame, pd.DataFrame):
        # -> build dataframe for simulation monitoring
        sim_data = pd.DataFrame(dict(iteration=[int(self.iteration)] * len(self.time_range),
                                     scenario=[self.scenario] * len(self.time_range),
                                     charged=self.charged,
                                     shifted=self.shifted,
                                     sub_id=self.sub_grid,
                                     price=self.prices + self.base_price,
                                     soc=self.soc,
                                     cost=self.cost,
                                     time=self.time_range))

        # -> build dataframe for reference car
        if self.reference_car:
            self.reference_car.monitor['time'] = self.time_range
            car_data = pd.DataFrame(self.reference_car.monitor)
            car_data['iteration'] = int(self.iteration)
            car_data['scenario'] = self.scenario
        else:
            car_data = None

        # -> determine car data
        evs, avg_demand, avg_distance = 0, 0, 0
        for household in self.clients.values():
            for person in household.persons:
                if person.car.type == 'ev':
                    evs += 1
                    avg_distance += person.car.odometer
                    if person.car.odometer > 0:
                        avg_demand += person.car.demand.sum() / person.car.odometer * 100
        days = len(self.time_range) / 1440

        sim_data['total_ev'] = evs
        sim_data['avg_distance'] = avg_distance/evs/days
        sim_data['avg_demand'] = avg_demand/evs

        return sim_data, car_data
