import numpy as np
import pandas as pd
from datetime import datetime

from participants.basic import BasicParticipant
from participants.utils import Resident
from demLib.electric_profile import StandardLoadProfile


class HouseholdModel(BasicParticipant):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # ---> initialize profile generator
        self.profile_generator = StandardLoadProfile(demandP=kwargs['demandP'], type='household', resolution='15min',
                                                     london_data=kwargs['london_data'], l_id=kwargs['l_id'])
        # ---> residents
        self.persons = [Resident(**kwargs) for i in range(kwargs['residents']) if i <= 1]

        self.delay = 0
        self.waiting_time = 0

    def get_fixed_power(self, d_time: datetime):
        return self.profile_generator.run_model(d_time)     # --> return time series (1/4 h) [kW]

    def get_request(self, d_time: datetime):
        if self.delay == 0:                                 # --> charging if waiting expired
            requests = [(0, 0)]
            for person in [p for p in self.persons if p.car.type == 'ev']:
                p, duration = person.car.plan_charging(d_time)
                requests += [(p, duration)]                 # --> add to request
            return {str(self.grid_node): requests}          # --> return to fp-agent
        elif self.delay > 0:
            self.delay -= 1
        return {str(self.grid_node): [(0, 0)]}

    def commit(self, price):
        if price < np.inf:
            free = any([p.price_limit > price for p in self.persons])
            require = any([p.car.soc < p.car.require for p in self.persons])
            if free or require:
                for person in self.persons:
                    if person.car.duration > 0:
                        person.car.charging = True
                    self.waiting_time = 0
                return True
        # --> wait 30-60 minutes till next try
        self.delay = np.random.randint(low=30, high=60)
        self.waiting_time += self.delay
        return False

    def simulate(self, d_time):
        for person in [p for p in self.persons if p.car.type == 'ev']:
            person.car.charge(d_time)           # --> perform charging
            demand = person.car.drive(d_time)   # --> drive the car
            if demand > 0:
                self.waiting_time = 0


if __name__ == "__main__":
    sim_paras = dict(T=96, start_date=pd.to_datetime('2022-01-01'), end_date=pd.to_datetime('2022-02-01'),
                     ev_ratio=1, minimum_soc=30, grid_node='nowum', residents=3, demandP=3000, london_data=False,
                     base_price=29)
    house = HouseholdModel(**sim_paras)
