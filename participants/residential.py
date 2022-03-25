import numpy as np
import pandas as pd
from datetime import datetime
import names
import logging

from participants.basic import BasicParticipant
from participants.utils import Resident
from demLib.electric_profile import StandardLoadProfile

# ---> price data from survey
mean_price = 28.01
var_price = 7.9


class HouseholdModel(BasicParticipant):

    def __init__(self, *args, **kwargs):
        super().__init__(**kwargs)
        # create family name
        self.family_name = names.get_last_name()
        # initialize profile generator
        self.profile_generator = StandardLoadProfile(demandP=kwargs['demandP'], type='household', resolution='15min',
                                                     random_choice=kwargs['london_data'])
        self.minimum_soc = kwargs['minimum_soc']

        # ---> create residents
        self.residents = []
        for num in range(kwargs['residents']):
            if num < 2:
                if np.random.choice(a=[True, False], p=[kwargs['employee_ratio'], 1 - kwargs['employee_ratio']]):
                    resident = Resident(['work', 'hobby', 'errand'], 'adult', self.family_name, **kwargs)
                    self.residents += [resident]
                else:
                    resident = Resident(['hobby', 'errand'], 'adult', self.family_name, **kwargs)
                    self.residents += [resident]
            else:
                self.residents += [Resident([], 'child', self.family_name, **kwargs)]

        # ---> price limits from survey
        self.price_low = round(np.random.normal(loc=mean_price, scale=var_price), 2)
        self.price_medium = 0.805 * self.price_low + 17.45
        self.price_limit = 1.1477 * self.price_medium + 1.51
        # ---> store all cars that are used
        self.car_manager = {resident.name: {'car': resident.car, 'request': 0, 'commit': False,
                                            'ask': True if resident.car.soc < self.minimum_soc else False,
                                            'user': resident, 'limit': 5}
                            for resident in self.residents
                            if resident.type == 'adult' and resident.car is not None and resident.car.type == 'ev'}
        self.delay = 0
        self.waiting_time = 0
        self.shift = False
        self.not_charged = False
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
            # ---> charging
            if manager['commit'] and manager['request'] > 0:                # ---> if commit and duration > 0
                manager['car'].charge()                                     # ---> perform charging
                manager['request'] -= 1                                     # ---> decrement charging time
            elif manager['commit']:                                         # ---> no charging time available
                manager['commit'] = False                                   # ---> charging done and reset
                manager['request'] = 0
            # ---> driving
            consumption = manager['car'].drive(d_time)                      # ---> drive the car
            if consumption > 0:                                             # ---> car moved -> new request
                self.shift = False                                          # ---> reset shift
                self.delay = 0                                              # ---> reset waiting time
                if manager['ask']:                                          # ---> no time for charging
                    self.not_charged = True                                 # ---> or to expensive
            # ---> charging required
            if manager['user'].car_usage.loc[d_time] == 0 \
                    and manager['car'].soc < self.minimum_soc and not manager['commit']:
                manager['ask'] = True                                       # ---> set ask to True
                self.not_charged = False                                    # --> reset not charged for new request

    def get_request(self, d_time: datetime):
        requests = []                                                       # ---> list for requests
        if self.delay == 0:                                                 # ---> charging if waiting expired
            for manager in [manager for manager in self.car_manager.values() if manager['ask']]:
                p, dt = manager['user'].plan_charging(d_time)               # ---> get planing
                if p > 0 and dt > 0:
                    requests += [(p, dt)]                                   # ---> add to request
                    manager['request'] = dt                                 # ---> set possible charging time
        elif self.delay > 0:                                                # ---> decrement waiting time
            self.delay -= 1
            self.waiting_time += 1
        if len(requests) > 0:
            return {str(self.grid_node): requests}                          # ---> return dictionary with requests
        else:
            return {}

    def commit_charging(self, price, d_time):
        for manager in self.car_manager.values():
            car_use = manager['user'].car_usage
            try:
                t1 = car_use.loc[(car_use.index > d_time) & (car_use == 1)].index[0]    # ---> get next usage
                t2 = car_use.loc[(car_use.index > t1) & (car_use == 0)].index[0]
                next_demand = (manager['car'].demand.loc[t1:t2].sum() / manager['car'].capacity) * 100
                next_demand = max(5, round(next_demand, 2))
                manager['limit'] = next_demand
            except Exception as e:
                self._logger.info(e)
                manager['limit'] = 5

        # ---> if the price is below the limit, commit the charging
        if price < self.price_limit or any([manager['car'].soc < manager['limit'] for manager in self.car_manager.values()]):
            waiting_time = 0
            for car_management in self.car_manager.values():                # ---> for each car
                if car_management['ask']:                                   # ---> check charging request
                    car_management['commit'] = True                         # ---> set commit flag
                    car_management['ask'] = False                           # ---> reset asking flag for charging time
            if self.shift:
                waiting_time = self.waiting_time                            # ---> return value for estimation
                self.waiting_time = 0                                       # ---> reset waiting time
                self.shift = False                                          # ---> reset shift flag
            return True, waiting_time
        else:
            self.delay = np.random.randint(low=30, high=60)                 # ---> wait 30-60 minutes till next try
            self.shift = True
            for car_management in self.car_manager.values():
                car_management['ask'] = False
            return False, 0


if __name__ == "__main__":
    house = HouseholdModel(T=96, demandP=3000, residents=3, grid_node='NOWUM', london_data=False, minimum_soc=30,
                           employee_ratio=0.7, ev_ratio=1, start_date=pd.to_datetime('2022-01-01'),
                           end_date=pd.to_datetime('2022-02-01'))
    power = house.get_fixed_power(pd.to_datetime('2022-01-01'))
    # a = house.residents[0].get_car_usage(pd.to_datetime('2018-01-01'))
    for x in pd.date_range(start='2022-01-01', periods=1440, freq='min'):
        house.do(x)

