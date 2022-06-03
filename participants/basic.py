import numpy as np
from datetime import timedelta as td
from datetime import datetime
import pandas as pd


class BasicParticipant:

    def __init__(self, T: int = 96, grid_node: str = None,
                 start_date: datetime = None, end_date: datetime = None, *args, **kwargs):

        # -> time resolution information
        self.T, self.t, self.dt = T, np.arange(T), 1/(T/24)
        self.time_range = pd.date_range(start=start_date, end=end_date + td(days=1), freq='min')[:-1]
        self.date_range = pd.date_range(start=start_date, end=end_date, freq='d')
        len_ = int(len(self.time_range)/15)
        # -> input parameters and time series for the optimization
        self.weather, self.prices = pd.DataFrame(index=self.time_range), pd.DataFrame(index=self.time_range)
        # -> optimization output time series
        # self.generation, self.demand = np.zeros(len_, float), np.zeros(len_, float)
        self._generation = pd.Series(index=self.time_range, dtype=float)
        self._demand = pd.Series(index=self.time_range, dtype=float)
        self._residual_demand = pd.Series(index=self.time_range, dtype=float)
        self._residual_generation = pd.Series(index=self.time_range, dtype=float)
        # -> grid connection node
        self.grid_node = grid_node

    def set_parameter(self, weather: pd.DataFrame = None, prices: pd.DataFrame = None):
        self.weather = weather
        self.prices = prices

    def optimize(self, d_time: datetime):
        pass