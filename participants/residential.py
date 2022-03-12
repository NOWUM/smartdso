import numpy as np
import pandas as pd
from datetime import datetime
import names
import os
import logging
from collections import defaultdict

from participants.basic import BasicParticipant
from participants.utils import Resident
from demLib.electric_profile import StandardLoadProfile

# ---> resident data
employee_ratio = os.getenv('EMPLOYEE_RATIO', 0.7)
london_data = (os.getenv('LONDON_DATA', 'False') == 'True')
minimum_soc = int(os.getenv('MINIMUM_SOC', 30))
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
        self.car_manager = {resident.name: {'car': resident.car, 'request': 0, 'commit': False,
                                            'ask': True if resident.car.soc < minimum_soc else False,
                                            'user': resident}
                            for resident in self.residents
                            if resident.type == 'adult' and resident.car is not None and resident.car.type == 'ev'}

        self.ask_energy = 0
        self.charging = []
        self.delay = 0
        self.shift = False
        self._logger = logging.getLogger(f'Household{self.family_name.capitalize()}')
        self._logger.setLevel('WARNING')

    def get_fixed_power(self, d_time: datetime):
        # ---> get standard load profile
        self.power = self.profile_generator.run_model(pd.to_datetime(d_time))
        self.demand['power'] = self.power
        return self.power                                                   # ---> return time series (1/4 h)

    def do(self, d_time: datetime):
        # ---> to action like charge and drive
        for manager in self.car_manager.values():
            if manager['commit'] and manager['request'] > 0:                # ---> if commit and duration > 0
                manager['car'].charge()                                     # ---> perform charging
                manager['request'] -= 1                                     # ---> decrement charging time
            elif manager['commit']:                                         # ---> no charging time available
                manager['commit'] = False                                   # ---> request done and reset
            manager['car'].drive(d_time)                                    # ---> move the car
            # ---> check if charging is required
            if manager['user'].car_usage.loc[d_time] == 0 and manager['car'].soc < minimum_soc:
                manager['ask'] = True                                       # ---> set ask to True

    def get_request(self, d_time: datetime):
        requests = []
        # ---> plan charging if the car is unused or the waiting time is expired
        if any([manager['ask'] for manager in self.car_manager.values()]) and self.delay == 0:
            self._logger.info(' ---> check for charging demand')
            for manager in self.car_manager.values():
                if manager['ask']:                                          # ---> if the current car need a charge
                    p, dt = manager['user'].plan_charging(d_time)           # ---> get planing
                    if p > 0 and dt > 0:
                        requests += [(p, dt)]                               # ---> add to request
                        manager['request'] = dt                             # ---> set as possible charging time
        # ---> if the waiting time is not expired decrement by one
        elif self.delay >= 0:
            if self.delay == 0:
                for manager in self.car_manager.values():
                    manager['ask'] = True                                   # ---> set asking to true
            else:
                self.delay -= 1
        if len(requests) > 0:
            return {str(self.grid_node): requests}                          # ---> return dictionary with requests
        else:
            return {}

    def commit_charging(self, price):
        # TODO: Check price --> current price per kWh is used
        # ---> if the price is below the limit, commit the charging
        if price < self.price_limit:
            shift = False
            for car_management in self.car_manager.values():
                car_management['commit'] = True                 # ---> set commit flag
                car_management['ask'] = False                   # ---> reset asking flag for charging time
            if self.shift:
                self.shift = False
                shift = True
            return True, shift                                  # ---> return True to commit
        else:
            self.delay = np.random.randint(low=60, high=120)    # ---> wait 60-120 minutes till next try
            self.shift = True
            return False, False                                 # ---> return False to reject


if __name__ == "__main__":
    house = HouseholdModel(T=96, demandP=3000, residents=3, grid_node='NOWUM')
    power = house.get_fixed_power(pd.to_datetime('2018-01-01'))
    # a = house.residents[0].get_car_usage(pd.to_datetime('2018-01-01'))
    # for x in pd.date_range(start='2018-01-01', periods=2, freq='min'):
    #    p = house.get_request(x)
