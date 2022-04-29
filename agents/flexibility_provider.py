import uuid
import pandas as pd
import numpy as np
from datetime import datetime, timedelta as td
from collections import defaultdict

from participants.residential import HouseholdModel
from participants.business import BusinessModel
from agents.utils import WeatherGenerator

# ---> read known consumers and nodes
allocated_consumers = pd.read_csv(r'./gridLib/data/grid_allocations.csv', index_col=0)
h0_consumers = allocated_consumers.loc[allocated_consumers['profile'] == 'H0']
g0_consumers = allocated_consumers.loc[allocated_consumers['profile'] == 'G0']
nodes = pd.read_csv(r'./gridLib/data/export/nodes.csv', index_col=0)

nuts_code = 'DEA26'


class FlexibilityProvider:

    def __init__(self, **kwargs):

        self.clients = {}               # --> total clients
        self.capacity = 0               # --> portfolio capacity
        self.power = 0                  # --> portfolio power
        # --> save time range
        self.time_range = pd.date_range(start=kwargs['start_date'], end=kwargs['end_date'] + td(days=1),
                                        freq='min')[:-1]
        self.indexer = 0
        len_ = len(self.time_range)
        self.dynamic_fee = kwargs['dynamic_fee']
        # --> simulation monitoring
        self.empty_counter = np.zeros(len_)         # --> counts the number of ev, which drive without energy
        self.virtual_source = np.zeros(len_)        # --> energy demand, which is needed if the ev is emtpy
        self.prices = np.zeros(len_)                # --> summarized prices of all nodes
        self.charged = np.zeros(len_)               # --> summarized charging of all cars
        self.shifted = np.zeros(len_)               # --> summarized shift charge of all cars
        self.sub_grid = -1*np.ones(len_)            # --> detect sub grid
        self.soc = np.zeros(len_)                   # --> summarized soc of all cars

        # --> create household clients
        for _, consumer in h0_consumers.iterrows():
            sim_parameters = dict(T=96, demandP=consumer['jeb'], grid_node=consumer['bus0'],
                                  residents=int(max(consumer['jeb'] / 1500, 1)))
            sim_parameters.update(kwargs)
            client = HouseholdModel(**sim_parameters)
            for person in [p for p in client.persons if p.car.type == 'ev']:
                self.capacity += person.car.capacity
                self.power += person.car.maximal_charging_power
            self.clients[uuid.uuid1()] = client
        # --> select reference car
        self.reference_car = None
        while self. reference_car is None:
            key = np.random.choice([key for key in self.clients.keys()])
            for person in [p for p in self.clients[key].persons if p.car.type == 'ev']:
                self.reference_car = person.car

        # --> create business clients
        for _, consumer in g0_consumers.iterrows():
            client = BusinessModel(T=96, demandP=consumer['jeb'], grid_node=consumer['bus0'], **kwargs)
            self.clients[uuid.uuid1()] = client

        # ---> set weather parameters
        self._nuts3 = nuts_code
        self._weather_generator = WeatherGenerator()

    def _generate_weather(self, d_time: datetime):
        weather = self._weather_generator.get_weather(d_time.replace(year=1996), self._nuts3)
        for participant in self.clients.values():
            participant.set_parameter(weather=weather.copy(), prices={})

    def get_fixed_power(self, d_time: datetime):
        total_powers = []
        for id_, participant in self.clients.items():
            df = pd.DataFrame({'power': participant.get_fixed_power(d_time)})
            df.index = pd.date_range(start=d_time, freq='15min', periods=len(df))
            df['id_'] = str(id_)
            df['node_id'] = participant.grid_node
            df = df.rename_axis('t')
            total_powers.append(df)

        return pd.concat(total_powers)

    def get_requests(self, d_time: datetime):
        requests = {id_: participant.get_request(d_time) for id_, participant in self.clients.items()}
        response = {}
        for id_, request in requests.items():
            for key, values in request.items():
                if any([duration > 0 for _, duration in values]):
                    response[id_] = request
        return response

    def simulate(self, d_time: datetime):
        capacity, empty, pool = 0, 0, 0
        for participant in self.clients.values():
            participant.simulate(d_time)
            for person in [p for p in participant.persons if p.car.type == 'ev']:
                capacity += person.car.soc / 100 * person.car.capacity
                empty += int(person.car.empty)
                pool += person.car.virtual_source
        self.soc[self.indexer] = (capacity / self.capacity) * 100
        # --> add to simulation monitoring
        self.empty_counter[self.indexer] = empty
        self.virtual_source[self.indexer] = pool
        # --> increment time counter
        self.indexer += 1

    def commit(self, id_, request: dict, price: float, sub_id: str):
        w = self.clients[id_].waiting_time
        if not self.dynamic_fee:
            price = 0
        if self.clients[id_].commit(price):
            self.prices[self.indexer] = price
            self.sub_grid[self.indexer] = int(sub_id)
            for _, parameters in request.items():
                for power, duration in parameters:
                    self.charged[self.indexer:self.indexer + duration] += power
                    if w > 0:
                        self.shifted[self.indexer:self.indexer + duration] += power
            return True
        else:
            return False

    def get_results(self):
        self.reference_car.monitor['time'] = self.time_range
        sim_data = pd.DataFrame(dict(charged=self.charged, shifted=self.shifted, sub_grid=self.sub_grid,
                                     price=self.prices, soc=self.soc, time=self.time_range))
        # --> determine car data
        evs, avg_demand, avg_distance = 0, 0, 0
        for household in self.clients.values():
            for person in household.persons:
                if person.car.type == 'ev':
                    evs += 1
                    avg_distance += person.car.odometer
                    if person.car.odometer > 0:
                        avg_demand += person.car.demand.sum() / person.car.odometer * 100
        days = len(self.time_range) / 1440
        car_data = pd.DataFrame(dict(evs=evs, distance=avg_distance/evs/days, demand=avg_demand/evs,
                                     time=self.time_range[0]), index=[0])

        return pd.DataFrame(self.reference_car.monitor), car_data, sim_data


if __name__ == "__main__":
    import os

    start_date = pd.to_datetime(os.getenv('START_DATE', '2022-01-01'))
    end_date = pd.to_datetime(os.getenv('END_DATE', '2022-01-02'))

    input_set = {'london_data': (os.getenv('LONDON_DATA', 'False') == 'True'),
                 'minimum_soc': int(os.getenv('MINIMUM_SOC', 50)),
                 'start_date': start_date,
                 'end_date': end_date,
                 'base_price': 29,
                 'ev_ratio': int(os.getenv('EV_RATIO', 100)) / 100}

    fp = FlexibilityProvider(**input_set)

