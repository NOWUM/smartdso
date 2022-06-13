import pandas as pd
import numpy as np
from datetime import datetime

from participants.basic import BasicParticipant
from demLib.electric_profile import StandardLoadProfile


class BusinessModel(BasicParticipant):

    def __init__(self,
                 demandP: float,
                 grid_node: str = None,
                 start_date: datetime = datetime(2022, 1, 1), end_date: datetime = datetime(2022, 1, 2),
                 T: int = 1440,
                 *args, **kwargs):
        super().__init__(T=T, grid_node=grid_node, start_date=start_date, end_date=end_date)
        self.profile_generator = StandardLoadProfile(demandP=demandP, type='business', resolution=self.T)

    def set_fixed_demand(self) -> None:
        # -> return time series (1/4 h) [kW]
        demand = np.asarray([self.profile_generator.run_model(date) for date in self.date_range]).flatten()
        self._demand = pd.Series(index=self.time_range, data=demand)

    def set_photovoltaic_generation(self) -> None:
        generation = np.zeros(96*len(self.date_range))
        if self.T == 1440:
            self._generation = pd.Series(index=self.time_range, data=np.repeat(generation, 15))
        elif self.T == 96:
            self._generation = pd.Series(index=self.time_range, data=generation)
        elif self.T == 24:
            generation = np.asarray([np.mean(generation[i:i + 3]) for i in range(0, 96, 4)], np.float).flatten()
            self._generation = pd.Series(index=self.time_range, data=generation)

    def set_residual(self) -> None:
        self._residual_demand = self._demand - self._generation
        self._residual_demand[self._residual_demand < 0] = 0
        self._residual_generation = self._generation - self._demand
        self._residual_generation[self._residual_generation < 0] = 0
