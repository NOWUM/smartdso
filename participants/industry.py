import os
from datetime import datetime

from participants.basic import BasicParticipant
from demLib.electric_profile import StandardLoadProfile

DATABASE_URI = os.getenv('DATABASE_URI', 'postgresql://opendata:opendata@10.13.10.41:5432/smartdso')


class IndustryModel(BasicParticipant):

    def __init__(self,
                 demandP: float,
                 grid_node: str = None,
                 start_date: datetime = datetime(2022, 1, 1), end_date: datetime = datetime(2022, 1, 2),
                 T: int = 1440, database_uri: str = DATABASE_URI,
                 *args, **kwargs):
        super().__init__(T=T, grid_node=grid_node, start_date=start_date, end_date=end_date,
                         database_uri=database_uri, consumer_type='industry', random=None)
        self.profile_generator = StandardLoadProfile(demandP=demandP, type='industry', resolution=self.T)
