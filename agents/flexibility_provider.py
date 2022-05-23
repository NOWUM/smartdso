import uuid
import pandas as pd
import numpy as np
from datetime import datetime, timedelta as td

from participants.residential import HouseholdModel
from participants.business import BusinessModel
from participants.industry import IndustryModel

# -> read known consumers and nodes
consumers = pd.read_csv(r'./gridLib/data/grid_allocations.csv', index_col=0)
h0_consumers = consumers.loc[consumers['profile'] == 'H0']
g0_consumers = consumers.loc[consumers['profile'] == 'G0']
rlm_consumers = consumers.loc[consumers['profile'] == 'RLM']


class FlexibilityProvider:

    def __init__(self, scenario: str, iteration: int, base_price: float, dynamic_fee: bool,
                 start_date: datetime, end_date: datetime, ev_ratio: float = 0.5,
                 minimum_soc: int = -1, london_data: bool = False, *args, **kwargs):

        # -> scenario name and iteration number
        self.scenario = scenario
        self.iteration = iteration
        # -> economic settings
        self.base_price = base_price
        self.dynamic_fee = dynamic_fee
        # -> total clients
        self.clients = {}
        # -> time range
        self.time_range = pd.date_range(start=start_date, end=end_date + td(days=1), freq='min')[:-1]
        self.indexer = 0
        len_ = len(self.time_range)
        # -> simulation monitoring
        self.capacity = 0
        self.power = 0
        self.empty_counter = np.zeros(len_)
        self.virtual_source = np.zeros(len_)
        self.prices = np.zeros(len_)
        self.charged = np.zeros(len_)
        self.shifted = np.zeros(len_)
        self.sub_grid = -1*np.ones(len_)
        self.soc = np.zeros(len_)
        self.cost = np.zeros(len_)

        # -> create household clients
        for _, consumer in h0_consumers.iterrows():

            client = HouseholdModel(demandP=consumer['jeb'],
                                    residents=int(max(consumer['jeb'] / 1500, 1)),
                                    london_data=london_data,
                                    l_id=consumer['london_data'],
                                    minimum_soc=minimum_soc,
                                    ev_ratio=ev_ratio,
                                    start_date=start_date,
                                    end_time=end_date,
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
            client = BusinessModel(T=96, demandP=consumer['jeb'], grid_node=consumer['bus0'], **kwargs)
            self.clients[uuid.uuid1()] = client

        # -> create industry clients
        for _, consumer in rlm_consumers.iterrows():
            client = IndustryModel(T=96, demandP=consumer['jeb'], grid_node=consumer['bus0'], **kwargs)
            self.clients[uuid.uuid1()] = client


    def _generate_weather(self, d_time: datetime):
        pass

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
        # -> add to simulation monitoring
        self.empty_counter[self.indexer] = empty
        self.virtual_source[self.indexer] = pool
        # -> increment time counter
        self.indexer += 1

    def commit(self, id_, request: dict, price: float, sub_id: str):
        w = self.clients[id_].waiting_time
        if not self.dynamic_fee:
            price = 0
        if self.clients[id_].commit(price):
            self.prices[self.indexer] = price
            self.sub_grid[self.indexer] = int(sub_id)
            energy = 0
            for _, parameters in request.items():
                for power, duration in parameters:
                    self.charged[self.indexer:self.indexer + duration] += power
                    if w > 0:
                        self.shifted[self.indexer:self.indexer + duration] += power
                    energy += (power * duration) / 60
            self.cost[self.indexer] += (energy * price)
            return True
        else:
            return False

    def get_results(self):
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
