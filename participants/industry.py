import pandas as pd
import numpy as np
from datetime import datetime

from participants.basic import BasicParticipant
from demLib.electric_profile import StandardLoadProfile


class IndustryModel(BasicParticipant):

    def __init__(self,
                 demandP: float,
                 grid_node: str = None,
                 start_date: datetime = datetime(2022, 1, 1), end_date: datetime = datetime(2022, 1, 2),
                 T: int = 1440,
                 *args, **kwargs):
        super().__init__(T=T, grid_node=grid_node, start_date=start_date, end_date=end_date)
        self.profile_generator = StandardLoadProfile(demandP=demandP, type='industry')

    def set_fixed_demand(self):
        # -> return time series (1/4 h) [kW]
        demand = np.asarray([self.profile_generator.run_model(date) for date in self.date_range]).flatten()
        self._demand = pd.Series(index=self.time_range, data=np.repeat(demand, 15))

    def set_photovoltaic_generation(self):
        generation = np.zeros(96*len(self.date_range))
        self._generation = pd.Series(index=self.time_range, data=np.repeat(generation, 15))

    def set_residual(self):
        self._residual_demand = self._demand - self._generation
        self._residual_demand[self._residual_demand < 0] = 0
        self._residual_generation = self._generation - self._demand
        self._residual_generation[self._residual_generation < 0] = 0