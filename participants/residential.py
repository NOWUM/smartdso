import numpy as np
import pandas as pd
from datetime import datetime
import names
import os
import logging

from participants.basic import BasicParticipant
from participants.utils import Resident
from demLib.electric_profile import StandardLoadProfile

# ---> resident data
employee_ratio = os.getenv('EMPLOYEE_RATIO', 0.7)
london_data = (os.getenv('LONDON_DATA', 'False') == 'True')

# ---> price data from survey
mean_price = 28.01
var_price = 7.9


class HouseholdModel(BasicParticipant):

    def __init__(self, T, *args, **kwargs):
        super().__init__(T, **kwargs)
        # create family name
        self.family_name = names.get_last_name()
        # initialize profile generator
        self.profile_generator = StandardLoadProfile(demandP=kwargs['demandP'], type='household', resolution='15min',
                                                     random_choice=london_data)
        # ---> create residents
        self.residents = []
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

        # ---> price limits from survey
        self.price_low = round(np.random.normal(loc=mean_price, scale=var_price), 2)
        self.price_medium = 0.805 * self.price_low + 17.45
        self.price_limit = 1.1477 * self.price_medium + 1.51
        # ---> store all cars that are used
        self.car_manager = {resident.name: {'car': resident.car, 'requests': pd.Series(dtype=float), 'commit': False}
                            for resident in self.residents if resident.type == 'adult'}
        self._electric = True if any([manager['car'].type == 'ev' for manager in self.car_manager.values()]) else False

        self.ask_energy = 0

        self.charging = []
        self.delay = 0

        self._logger = logging.getLogger(f'Household{self.family_name.capitalize()}')
        self._logger.setLevel('WARNING')

    def get_fixed_demand(self, d_time: datetime):
        # ---> get standard load profile
        self.demand['power'] = self.profile_generator.run_model(pd.to_datetime(d_time))
        self.power = self.demand['power']
        return self.power

    def do(self, d_time: datetime):
        for car_management in self.car_manager.values():
            if car_management['commit'] and sum(car_management['requests'].index > d_time) > 0:
                car_management['car'].charge()
            elif car_management['commit']:
                car_management['commit'] = False
            car_management['car'].drive(d_time)

    def get_request(self, d_time: datetime):
        # ---> plan charging if the car is unused or the waiting time is expired
        if self._electric and len(self.charging) == 0 and self.delay == 0:
            self._logger.info('check for charging demand')
            total_request = pd.Series(dtype=float)
            for resident in self.residents:
                if resident.own_car and resident.car.type == 'ev':
                    request = resident.plan_charging(d_time)
                    self.car_manager[resident.name]['requests'] = request
                    total_request = total_request.add(request, fill_value=0)
            total_request['node_id'] = self.grid_node
            self.ask_energy = total_request.sum() / 60
            return total_request
        elif self.delay > 0:
            self.delay -= 1
            return pd.Series()
        else:
            return pd.Series()

    def commit_charging(self, price):
        average_price = price / self.ask_energy
        if average_price < self.price_limit:
            for car_management in self.car_manager.values():
                car_management['commit'] = True
            return True
        else:
            self.delay = np.random.randint(low=15, high=30)
            return False


if __name__ == "__main__":
    house = HouseholdModel(T=96, demandP=3000, residents=3, grid_node='NOWUM')
    power = house.get_fixed_demand(pd.to_datetime('2018-01-01'))
    # a = house.residents[0].get_car_usage(pd.to_datetime('2018-01-01'))
    #for x in pd.date_range(start='2018-01-01', periods=2, freq='min'):
    #    p = house.get_request(x)

