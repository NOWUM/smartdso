import numpy as np
from datetime import timedelta as td
from datetime import datetime
import pandas as pd

RESOLUTION = {1440: 'min', 96: '15min', 24: 'h'}


class BasicParticipant:

    def __init__(self, T: int = 1440,
                 grid_node: str = None,
                 start_date: datetime = None, end_date: datetime = None,
                 *args, **kwargs):

        # -> time resolution information
        self.T, self.t, self.dt = T, np.arange(T), 1/(T/24)
        self.time_range = pd.date_range(start=start_date, end=end_date + td(days=1), freq=RESOLUTION[self.T])[:-1]
        self.date_range = pd.date_range(start=start_date, end=end_date, freq='d')
        # -> input parameters and time series for the optimization
        self.weather, self.prices = pd.DataFrame(index=self.time_range), pd.DataFrame(index=self.time_range)
        # -> optimization output time series
        steps = len(self.time_range)
        self._generation = pd.Series(index=self.time_range, data=np.zeros(steps))
        self._demand = pd.Series(index=self.time_range, data=np.zeros(steps))
        self._residual_demand = pd.Series(index=self.time_range, data=np.zeros(steps))
        self._residual_generation = pd.Series(index=self.time_range, data=np.zeros(steps))
        self.power = pd.Series(index=self.time_range, data=np.zeros(steps))
        # -> grid connection node
        self.grid_node = grid_node
        self.persons = []

    def set_parameter(self, weather: pd.DataFrame = None, prices: pd.DataFrame = None) -> None:
        self.weather = weather
        self.prices = prices

    def simulate(self, d_time: datetime) -> None:
        pass

    def get_request(self, d_time: datetime) -> pd.Series:
        self.power = pd.Series(data=np.zeros(self.T), index=pd.date_range(start=d_time, periods=self.T, freq='min'))
        return self.power

    def get_demand(self) -> (pd.Series, pd.Series):
        return self._demand, self._residual_demand

    def get_generation(self) -> (pd.Series, pd.Series):
        return self._generation, self._residual_generation


