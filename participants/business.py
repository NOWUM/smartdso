from datetime import datetime

import numpy as np

from demLib.electric_profile import StandardLoadProfile
from participants.basic import BasicParticipant


class BusinessModel(BasicParticipant):
    def __init__(
        self,
        demandP: float,
        database_uri: str,
        grid_node: str = None,
        start_date: datetime = datetime(2022, 1, 1),
        end_date: datetime = datetime(2022, 1, 2),
        steps: int = 96,
        resolution: str = '15min',
        *args,
        **kwargs
    ):
        super().__init__(
            steps=steps,
            resolution=resolution,
            grid_node=grid_node,
            start_date=start_date,
            end_date=end_date,
            database_uri=database_uri,
            consumer_type="business",
            random=None,
            pv_systems=[]
        )

        self._profile_generator = StandardLoadProfile(
            demandP=demandP, type="business", resolution=self.T
        )
