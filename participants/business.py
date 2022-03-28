import pandas as pd
from datetime import datetime

from participants.basic import BasicParticipant
from demLib.electric_profile import StandardLoadProfile


class BusinessModel(BasicParticipant):

    def __init__(self, T, *args, **kwargs):
        super().__init__(T, **kwargs)
        self.profile_generator = StandardLoadProfile(demandP=kwargs['demandP'], type='business', resolution='15min',
                                                     random_choice=False)
        self.persons = []

    def get_fixed_power(self, d_time: datetime):
        # ---> get standard load profile
        self.demand['power'] = self.profile_generator.run_model(pd.to_datetime(d_time))
        self.power = self.demand['power']
        return self.power

    def simulate(self, d_time: datetime):
        pass

    def get_request(self, d_time: datetime):
        return {str(self.grid_node): [(0, 0)]}

    def commit_charging(self, price):
        return False

    def set_mobility(self, d_time: datetime):
        pass