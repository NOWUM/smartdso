import numpy as np
from datetime import datetime

from participants.basic import BasicParticipant
from participants.utils import Resident
from demLib.electric_profile import StandardLoadProfile


class HouseholdModel(BasicParticipant):

    def __init__(self, demandP: float, residents: int, london_data: bool = False, l_id: str = None,
                 ev_ratio = 0.5, minimum_soc=-1, start_date=datetime(2022,1,1), end_time=datetime(2022,1,2),
                 base_price: float = 29, T: int = 1440, grid_node: str = None,*args, **kwargs):
        super().__init__(T=T, grid_node=grid_node)

        # -> initialize profile generator
        self.profile_generator = StandardLoadProfile(demandP=demandP, london_data=london_data, l_id=l_id)
        # -> create residents with cars
        self.persons = [Resident(ev_ratio=ev_ratio, minimum_soc=minimum_soc, base_price=base_price,
                                 start_date=start_date, end_time=end_time)
                        for _ in range(min(2, residents))]

        self.delay = 0
        self.waiting_time = 0

    def get_fixed_power(self, d_time: datetime):
        # -> return time series (1/4 h) [kW]
        return self.profile_generator.run_model(d_time)

    def get_request(self, d_time: datetime):
        # -> charging if waiting expired
        if self.delay == 0:
            requests = [(0, 0)]
            for person in [p for p in self.persons if p.car.type == 'ev']:
                p, duration = person.car.plan_charging(d_time)
                requests += [(p, duration)]                 # -> add to request
            return {str(self.grid_node): requests}          # -> return to fp-agent
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
        # -> wait 30-60 minutes till next try
        self.delay = np.random.randint(low=30, high=60)
        self.waiting_time += self.delay
        return False

    def simulate(self, d_time):
        for person in [p for p in self.persons if p.car.type == 'ev']:
            person.car.charge(d_time)           # -> do charging
            demand = person.car.drive(d_time)   # -> do driving
            if demand > 0:
                self.waiting_time = 0


if __name__ == "__main__":
    house = HouseholdModel(residents=5, demandP=5000)
