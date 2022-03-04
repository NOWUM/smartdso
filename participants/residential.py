import numpy as np
import pandas as pd
from datetime import datetime
import names
import os
import logging

from participants.basic import BasicParticipant
from participants.utils import Resident, in_minutes
from demLib.electric_profile import StandardLoadProfile
from collections import defaultdict

employee_ratio = os.getenv('EMPLOYEE_RATIO', 0.7)
london_data = (os.getenv('LONDON_DATA', 'False') == 'True')

mean_price = 28.01
var_price = 7.9


class HouseholdModel(BasicParticipant):

    def __init__(self, T, *args, **kwargs):
        super().__init__(T, **kwargs)

        self.profile_generator = StandardLoadProfile(demandP=kwargs['demandP'], type='household', resolution='min',
                                                     random_choice=london_data)
        self.residents = []

        self.family_name = names.get_last_name()
        for num in range(kwargs['residents']):
            if num < 2:
                if np.random.choice(a=[True, False], p=[employee_ratio, 1 - employee_ratio]):
                    resident = Resident(['work', 'hobby', 'errand'], 'adult', self.family_name)
                    self.residents += [resident]
                else:
                    resident = Resident(['hobby', 'errand'], 'adult', self.family_name)
                    self.residents += [resident]
            else:
                self.residents += [Resident([], 'child', self.family_name)]

        self.price_low = round(np.random.normal(loc=mean_price, scale=var_price), 2)
        self.price_medium = 0.805 * self.price_low + 17.45
        self.price_limit = 1.1477 * self.price_medium + 1.51
        self.last_request = None

        self.charging = []
        self.delay = 0

        self._logger = logging.getLogger(f'Household{self.family_name.capitalize()}')
        self._logger.setLevel('WARNING')

    def get_fixed_demand(self, d_time: datetime):
        # ---> get standard load profile
        self.demand['power'] = self.profile_generator.run_model(pd.to_datetime(d_time))
        # ---> add charging power if available from day before
        for power in self.charging:
            self.demand['power'] += power
        self.power = self.demand['power']
        return self.power

    def set_mobility(self, d_time: datetime):
        for resident in self.residents:
            resident.get_car_usage(d_time)

    def move(self, d_time: datetime):
        for resident in self.residents:
            if resident.own_car and resident.car.type == 'ev':
                resident.car.drive(in_minutes(d_time))

    def get_request(self, d_time: datetime):
        power = defaultdict(float)

        if len(self.charging) == 0 and self.delay == 0:
            # ---> plan a charging process
            self._logger.info('check for charging demand')
            require_charge = 0
            for resident in self.residents:
                if resident.own_car and resident.car.type == 'ev':
                    t1, t2 = resident.plan_charging(d_time)
                    for t in range(t1, t2):
                        power[t] += 22
                        require_charge += 1
            if require_charge > 0:
                self._logger.info('requiring charging ---> prepared request')
                df = pd.DataFrame.from_dict({'power': power}, orient='columns')
                df['node_id'] = self.grid_node
                df['t'] = df.index
                self.last_request = df.set_index(['node_id', 't'])
                return self.last_request

        elif len(self.charging) > 0:
            # ---> charge the car
            for resident in self.residents:
                if resident.own_car and resident.car.type == 'ev':
                    resident.car.charge()
            self.charging.pop()

        elif self.delay > 0:
            # ---> waiting before for new charging process
            self.delay -= 1

        return pd.DataFrame()

    def commit_charging(self, price):
        mean_price = price/(self.last_request['power'].sum()/60)
        if mean_price < self.price_limit:
            t1 = self.last_request.index.get_level_values('t').min()
            t2 = self.last_request.index.get_level_values('t').max()
            self.charging = list(self.last_request['power'].values)
            return True
        else:
            self.delay = np.random.randint(low=15, high=30)
            return False


if __name__ == "__main__":
    house = HouseholdModel(T=1440, demandP=3000, residents=3, grid_node='NOWUM')
    house.get_fixed_demand(pd.to_datetime('2018-01-01'))
    # a = house.residents[0].get_car_usage(pd.to_datetime('2018-01-01'))
    for x in pd.date_range(start='2018-01-01', periods=2, freq='min'):
        p = house.get_request(x)

