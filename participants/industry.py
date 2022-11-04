from datetime import datetime

import pandas as pd

from demLib.electric_profile import StandardLoadProfile
from participants.basic import BasicParticipant


class IndustryModel(BasicParticipant):
    def __init__(
        self,
        demandP: float,
        database_uri: str,
        grid_node: str = None,
        start_date: datetime = datetime(2022, 1, 1),
        end_date: datetime = datetime(2022, 1, 2),
        steps: int = 96,
        resolution: str = '15min',
        weather: pd.DataFrame = None,
        tariff: pd.Series = None,
        grid_fee: pd.Series = None,
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
            consumer_type="industry",
            random=None,
            pv_systems=[],
            weather=weather,
            tariff=tariff,
            grid_fee=grid_fee
        )

        self.profile_generator = StandardLoadProfile(
            demandP=demandP, type="industry", resolution=self.T
        )
