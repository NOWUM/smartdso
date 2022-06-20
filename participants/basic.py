import numpy as np
import os
from datetime import timedelta as td
from datetime import datetime
import pandas as pd
from sqlalchemy import create_engine

DATABASE_URI = os.getenv('DATABASE_URI', 'postgresql://opendata:opendata@10.13.10.41:5432/smartgrid')
RESOLUTION = {1440: 'min', 96: '15min', 24: 'h'}


class BasicParticipant:

    def __init__(self, T: int = 1440,
                 grid_node: str = None,
                 start_date: datetime = None, end_date: datetime = None,
                 database_uri: str = DATABASE_URI,
                 *args, **kwargs):

        # -> time resolution information
        self.T, self.t, self.dt = T, np.arange(T), 1/(T/24)
        self.time_range = pd.date_range(start=start_date, end=end_date + td(days=1), freq=RESOLUTION[self.T])[:-1]
        self._date_range = pd.date_range(start=start_date, end=end_date, freq='d')
        # -> input parameters and time series for the optimization
        self.weather, self.prices = pd.DataFrame(index=self.time_range), pd.DataFrame(index=self.time_range)
        # -> optimization output time series
        self._steps = len(self.time_range)
        self._data = pd.DataFrame(columns=['consumer_id', 'demand', 'residual_demand',
                                           'generation', 'residual_generation', 'pv_capacity',
                                           'car_capacity', 'total_radiation', 'tariff', 'grid_fee',
                                           'car_demand', 'planed_grid_consumption',
                                           'final_grid_consumption', 'planed_pv_consumption',
                                           'final_pv_consumption'], index=self.time_range)
        # -> grid connection node
        self.grid_node = grid_node
        self.persons = []

        self._database = create_engine(database_uri)

        self._request = pd.Series(dtype=float)

        self._finished = False
        self._initial_plan = False
        self._commit = self.time_range[0] - td(minutes=1)

    def has_commit(self):
        return self._finished

    def reset_commit(self):
        self._finished = False
        self._initial_plan = False

    def set_parameter(self, weather: pd.DataFrame = None, prices: pd.DataFrame = None) -> None:
        self.weather = weather
        self.prices = prices
        self._data.loc[self.time_range, 'total_radiation'] = self.weather['ghi'].values.flatten()

    def simulate(self, d_time: datetime) -> None:
        pass

    def get_request(self, d_time: datetime, strategy: str = None) -> pd.Series:
        if d_time > self._charging:
            steps = range(min(self.T, len(self.power.loc[d_time:])))
            self.power = pd.Series(data=np.zeros(self.T),
                                   index=pd.date_range(start=d_time, periods=self.T,
                                                       freq=RESOLUTION[self.T]))
            self._charging = self.power.index[-1]

        return self.power

    def get_demand(self) -> (pd.Series, pd.Series):
        return self._demand, self._residual_demand

    def get_generation(self) -> (pd.Series, pd.Series):
        return self._generation, self._residual_generation


