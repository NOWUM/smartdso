import uuid
import os
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta as td
from sqlalchemy import create_engine
from collections import defaultdict


from participants.residential import HouseholdModel
from participants.business import BusinessModel
from participants.industry import IndustryModel
from agents.utils import WeatherGenerator

SEED = int(os.getenv('RANDOM_SEED', 2022))
random = np.random.default_rng(SEED)

# -> read known consumers and nodes
consumers = pd.read_csv(r'./gridLib/data/export/dem/consumers.csv', index_col=0)
h0_consumers = consumers.loc[consumers['profile'] == 'H0']  # -> all h0 consumers
h0_consumers = h0_consumers.fillna(0)  # -> without pv = 0
g0_consumers = consumers.loc[consumers['profile'] == 'G0']  # -> all g0 consumers
rlm_consumers = consumers.loc[consumers['profile'] == 'RLM']  # -> all rlm consumers

# -> pandas frequency names
RESOLUTION = {1440: 'min', 96: '15min', 24: 'h'}
# -> database uri to store the results
DATABASE_URI = os.getenv('DATABASE_URI', 'postgresql://opendata:opendata@10.13.10.41:5432/smartdso')

logger = logging.getLogger('FlexibilityProvider')


class FlexibilityProvider:

    def __init__(self, scenario: str, iteration: int,
                 start_date: datetime, end_date: datetime, ev_ratio: float = 0.5,
                 london_data: bool = False, pv_ratio: float = 0.3, T: int = 1440,
                 database_uri: str = DATABASE_URI,
                 number_consumers: int = 0,
                 price_sensitivity: float = 1.3, strategy: str = 'MaxPvCap', *args, **kwargs):

        # -> scenario name and iteration number
        self.scenario = scenario
        self.iteration = iteration
        self.strategy = strategy
        # -> total clients
        self.clients = {}
        # -> weather generator
        # self.weather_generator = WeatherGeneratorDB()
        self.weather_generator = WeatherGenerator()
        # -> time range
        self.time_range = pd.date_range(start=start_date, end=end_date + td(days=1), freq=RESOLUTION[T])[:-1]

        self._database = create_engine(database_uri)

        self.T = T

        self.random = np.random.default_rng(iteration)

        global h0_consumers

        if number_consumers > 0:
            h0_consumers = h0_consumers.sample(number_consumers)

        # -> create household clients
        for _, consumer in h0_consumers.iterrows():
            # -> check pv potential and add system corresponding to the pv ratio
            if consumer['pv'] == 0:
                pv_systems = []
            elif self.random.choice(a=[True, False], p=[pv_ratio, 1 - pv_ratio]):
                pv_systems = eval(consumer['pv'])
            else:
                pv_systems = []

            # -> initialize h0 consumers
            id_ = uuid.uuid1()
            client = HouseholdModel(demandP=consumer['jeb'], consumer_id=str(id_), grid_node=consumer['bus0'],
                                    residents=int(max(consumer['jeb'] / 1500, 1)), ev_ratio=ev_ratio,
                                    london_data=london_data, l_id=consumer['london_data'],
                                    pv_systems=pv_systems, random=random,
                                    price_sensitivity=price_sensitivity,
                                    strategy=self.strategy, scenario=scenario,
                                    start_date=start_date, end_date=end_date, T=T,
                                    database_uri=database_uri, consumer_type='household')

            self.clients[id_] = client

        # -> create business clients
        for _, consumer in g0_consumers.iterrows():
            client = BusinessModel(T=T, demandP=consumer['jeb'], grid_node=consumer['bus0'],
                                   start_date=start_date, end_date=end_date, consumer_type='business')
            self.clients[uuid.uuid1()] = client

        # -> create industry clients
        for _, consumer in rlm_consumers.iterrows():
            client = IndustryModel(T=T, demandP=consumer['jeb'], grid_node=consumer['bus0'],
                                   start_date=start_date, end_date=end_date, consumer_type='industry')
            self.clients[uuid.uuid1()] = client

        self.keys = [key for key, value in self.clients.items() if value.consumer_type == 'household']
        self._commits = {key: False for key in self.keys}

    def initialize_time_series(self) -> (pd.DataFrame, pd.DataFrame):

        def build_dataframe(data, id_):
            dataframe = pd.DataFrame({'power': data.values})
            dataframe.index = self.time_range
            dataframe['id_'] = str(id_)
            dataframe['node_id'] = client.grid_node
            dataframe = dataframe.rename_axis('t')
            return dataframe

        weather = pd.concat([self.weather_generator.get_weather(date=date)
                             for date in pd.date_range(start=self.time_range[0], end=self.time_range[-1] + td(days=1),
                                                       freq='d')])
        weather = weather.resample('15min').ffill()
        weather = weather.loc[weather.index.isin(self.time_range)]

        demand_, generation_ = [], []
        for id_, client in self.clients.items():
            client.set_parameter(weather=weather.copy())
            client.initial_time_series()
            _, demand = client.get_demand()
            demand_.append(build_dataframe(demand, id_))
            _, generation = client.get_generation()
            generation_.append(build_dataframe(generation, id_))

        return pd.concat(demand_), pd.concat(generation_)

    def get_commits(self) -> int:
        return sum([int(c) for c in self._commits.values()])

    def get_requests(self, d_time: datetime) -> (pd.Series, str):
        random.shuffle(self.keys)
        for id_ in self.keys:
            self._commits[id_] = self.clients[id_].has_commit()
            if not self._commits[id_]:
                request = self.clients[id_].get_request(d_time, strategy=self.strategy)
                if sum(request.values) > 0:
                    yield request, self.clients[id_].grid_node, id_

    def simulate(self, d_time: datetime) -> None:
        capacity, empty, pool = 0, 0, 0
        for participant in self.clients.values():
            participant.simulate(d_time)
            for person in [p for p in participant.persons if p.car.type == 'ev']:
                capacity += person.car.soc * person.car.capacity
                empty += int(person.car.empty)
                pool += person.car.virtual_source

    def commit(self, price: pd.Series, consumer_id: uuid.uuid1) -> bool:
        commit_ = self.clients[consumer_id].commit(price=price)
        if commit_:
            self._commits[consumer_id] = self.clients[consumer_id].has_commit()
        return commit_

    def _save_summary(self, d_time: datetime) -> None:

        time_range = pd.date_range(start=d_time, freq=RESOLUTION[self.T], periods=self.T)

        result = pd.DataFrame(index=time_range, columns=['initial_grid', 'final_grid', 'final_pv',
                                                         'demand', 'residual_generation', 'availability',
                                                         'grid_fee'])
        result = result.fillna(0)

        consumer_counter, car_counter = 0, 0

        for id_, client in self.clients.items():
            if client.consumer_type == 'household':
                consumer_counter += 1
                # -> reset parameter for optimization for the next day
                client.reset_commit()
                self._commits[id_] = False
                # -> get results for current day
                data = client.get_result(time_range)
                # -> collect results
                result.loc[time_range, 'initial_grid'] += data.loc[time_range, 'planned_grid_consumption']
                result.loc[time_range, 'final_grid'] += data.loc[time_range, 'final_grid_consumption']
                result.loc[time_range, 'final_pv'] += data.loc[time_range, 'final_pv_consumption']
                result.loc[time_range, 'demand'] += data.loc[time_range, 'demand']
                result.loc[time_range, 'residual_generation'] += data.loc[time_range, 'residual_generation']
                result.loc[time_range, 'grid_fee'] += data.loc[time_range, 'grid_fee']

                for key, car in client.cars.items():
                    car_counter += 1
                    data = car.get_result(time_range)
                    result['availability'].loc[time_range] += (1-data.loc[time_range, 'usage'])

        result['availability'] /= car_counter
        result['grid_fee'] /= consumer_counter

        result['scenario'] = self.scenario
        result['iteration'] = self.iteration

        result.index.name = 'time'
        result = result.reset_index()
        result = result.set_index(['time', 'scenario', 'iteration'])

        try:
            result.to_sql(name='charging_summary', con=self._database, if_exists='append', method='multi')
        except Exception as e:
            logger.warning(f'server closed the connection {repr(e)}')

    def _save_electric_vehicles(self, d_time: datetime):
        time_range = pd.date_range(start=d_time, freq=RESOLUTION[self.T], periods=self.T)

        for id_, client in self.clients.items():
            for key, car in client.cars.items():
                data = car.get_result(time_range)
                data = data.loc[:, ['soc', 'usage', 'planned_charge', 'final_charge', 'demand', 'distance']]
                data.columns = ['soc', 'usage', 'initial_charging', 'final_charging', 'demand', 'distance']
                data['scenario'] = self.scenario
                data['iteration'] = self.iteration
                data['id_'] = key
                data.index.name = 'time'
                data = data.reset_index()
                data = data.set_index(['time', 'scenario', 'iteration', 'id_'])

                try:
                    data.to_sql(name='electric_vehicle', con=self._database, if_exists='append', method='multi')
                except Exception as e:
                    logger.warning(f'server closed the connection {repr(e)}')

    def save_results(self, d_time: datetime) -> None:
        self._save_summary(d_time)
        if self.iteration == 0:
            self._save_electric_vehicles(d_time)
