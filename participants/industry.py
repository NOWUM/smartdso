import os
from datetime import datetime
import numpy as np

from participants.basic import BasicParticipant
from demLib.electric_profile import StandardLoadProfile

from config import DATABASE_URI


class IndustryModel(BasicParticipant):

    def __init__(self,
                 demandP: float,
                 grid_node: str = None,
                 start_date: datetime = datetime(2022, 1, 1), end_date: datetime = datetime(2022, 1, 2),
                 T: int = 1440, database_uri: str = DATABASE_URI,
                 *args, **kwargs):
        super().__init__(T=T, grid_node=grid_node, start_date=start_date, end_date=end_date,
                         database_uri=database_uri, consumer_type='industry', random=None)
        self._data.loc[self.time_range, 'grid_fee'] = np.random.normal(2.6, 1e-6, self._steps)
        self.profile_generator = StandardLoadProfile(demandP=demandP, type='industry', resolution=self.T)
