import numpy as np
import os
from datetime import timedelta as td
from datetime import datetime
import pandas as pd
from sqlalchemy import create_engine


from demLib.electric_profile import StandardLoadProfile

DATABASE_URI = os.getenv('DATABASE_URI', 'postgresql://opendata:opendata@10.13.10.41:5432/smartgrid')
RESOLUTION = {1440: 'min', 96: '15min', 24: 'h'}


class BasicParticipant:

    def __init__(self, T: int = 1440, grid_node: str = None,
                 start_date: datetime = None, end_date: datetime = None,
                 database_uri: str = DATABASE_URI, consumer_type='household',
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
                                           'car_demand', 'planned_grid_consumption',
                                           'final_grid_consumption', 'planned_pv_consumption',
                                           'final_pv_consumption'], index=self.time_range)

        for column in self._data.columns:
            self._data[column] = np.zeros(len(self.time_range))

        # -> grid connection node
        self.grid_node = grid_node
        self.persons = []

        self.consumer_type = consumer_type

        self._profile_generator = StandardLoadProfile(demandP=1000, type=consumer_type, resolution=self.T)

        self._database = create_engine(database_uri)

        self._request = pd.Series(dtype=float)

        self._finished, self._initial_plan = False, True
        self._commit = self.time_range[0] - td(minutes=1)
        self._pv_systems = []
        self.cars = {}

    def initial_time_series(self):
        # -> return time series (1/4 h) [kW]
        demand = np.hstack([self._profile_generator.run_model(date) for date in self._date_range])
        self._data.loc[self.time_range, 'demand'] = demand
        # -> calculate generation
        generation = np.zeros(self._steps)
        for system in self._pv_systems:
            # -> irradiance unit [W/m²]
            rad_ = system.get_irradiance(solar_zenith=self.weather['zenith'], solar_azimuth=self.weather['azimuth'],
                                         dni=self.weather['dni'], ghi=self.weather['ghi'], dhi=self.weather['dhi'])
            # -> get generation in [kW/m^2] * [m^2]
            power = rad_['poa_global'] / 1e3 * system.arrays[0].module_parameters['pdc0']
            generation += power.values

        if self.T == 1440:
            self._data.loc[self.time_range, 'generation'] = np.repeat(generation, 15)
        elif self.T == 96:
            self._data.loc[self.time_range, 'generation'] = generation
        elif self.T == 24:
            generation = np.asarray([np.mean(generation[i:i + 3]) for i in range(0, 96, 4)], np.float).flatten()
            self._data.loc[self.time_range, 'generation'] = generation
        # -> set residual time series
        residual_demand = self._data['demand'] - self._data['generation']
        residual_demand[residual_demand < 0] = 0
        self._data.loc[self.time_range, 'residual_demand'] = residual_demand

        residual_generation = self._data['generation'] - self._data['demand']
        residual_generation[residual_generation < 0] = 0
        self._data.loc[self.time_range, 'residual_generation'] = residual_generation

        self._data.loc[self.time_range, 'car_demand'] = np.zeros(self._steps)
        for car in self.cars.values():
            self._data.loc[self.time_range, 'car_demand'] += car.get_data('demand').values

    def has_commit(self) -> bool:
        return self._finished

    def reset_commit(self) -> None:
        self._finished = False
        self._initial_plan = False

    def set_parameter(self, weather: pd.DataFrame = None, prices: pd.DataFrame = None) -> None:
        self.weather = weather
        self.prices = prices
        self._data.loc[self.time_range, 'total_radiation'] = weather['ghi'].values.flatten()

    def simulate(self, d_time):
        for person in [p for p in self.persons if p.car.type == 'ev']:
            person.car.charge(d_time)   # -> do charging
            person.car.drive(d_time)    # -> do driving

    def get_request(self, d_time: datetime, strategy: str = None) -> pd.Series:
        if d_time > self._commit:
            self._commit = d_time + td(days=1)
            self._finished = True
            self._initial_plan = True
        return pd.Series(dtype=float, index=[d_time], data=[0])

    def get_demand(self, time_range=None) -> (pd.Series, pd.Series):
        if time_range is None:
            time_range = self.time_range
        return self._data.loc[time_range, 'demand'], self._data.loc[time_range, 'residual_demand']

    def get_generation(self, time_range=None) -> (pd.Series, pd.Series):
        if time_range is None:
            time_range = self.time_range
        return self._data.loc[time_range, 'generation'], self._data.loc[time_range, 'residual_generation']

    def get_result(self, time_range: pd.DatetimeIndex = None) -> pd.DataFrame:
        if time_range is None:
            time_range = self.time_range
        return self._data.loc[time_range]