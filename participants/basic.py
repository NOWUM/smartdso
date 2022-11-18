import os
from datetime import datetime
from datetime import timedelta as td
from enum import Enum

import numpy as np
import pandas as pd
from pvlib.irradiance import get_total_irradiance
from sqlalchemy import create_engine

from carLib.car import Car, CarData
from demLib.electric_profile import StandardLoadProfile as PowerProfile
from demLib.heat_profile import StandardLoadProfile as HeatProfile
from participants.resident import Resident


class DataType(Enum):
    generation = "generation"
    residual_generation = "residual_generation"
    demand = "demand"
    residual_demand = "residual_demand"
    planned_pv_consumption = "planned_pv_consumption"
    final_pv_consumption = "final_pv_consumption"
    planned_grid_consumption = "planned_grid_consumption"
    final_grid_consumption = "final_grid_consumption"
    car_capacity = "car_capacity"
    grid_fee = "grid_fee"
    car_demand = "car_demand"


def resample_time_series(series: np.array, steps):
    func = {96: lambda x: x,
            1440: lambda x: np.repeat(x, 15),
            60: lambda x: np.array([np.mean(x[i:i+4]) for i in range(0, 96, 4)])}
    return func[steps](series)


class BasicParticipant:
    def __init__(
        self,
        database_uri: str,
        random: np.random.default_rng,
        name: str = "testing",
        sim: int = 0,
        steps: int = 96,
        resolution: str = "15min",
        grid_node: str = None,
        start_date: datetime = None,
        end_date: datetime = None,
        consumer_type: str = "household",
        consumer_id: str = "nowum",
        strategy: str = "optimized",
        profile_generator: dict = None,
        pv_systems: list = None,
        sub_grid: int = -1,
        residents: int = 0,
        weather: pd.DataFrame = None,
        tariff: pd.Series = None,
        grid_fee: pd.Series = None,
        *args,
        **kwargs
    ):

        # -> connection to database
        # self._database = create_engine(database_uri)
        # -> grid connection node
        self.grid_node = grid_node
        self.sub_grid = sub_grid
        # -> number of persons/residents
        self.residents = residents
        # -> drivers and cars
        self.drivers: list[Resident] = []
        self.cars: dict[str, Car] = {}
        # -> consumer identifier and type
        self.id_ = consumer_id
        self.consumer_type = consumer_type
        # -> set scenario name and iteration number
        self.name = name
        self.sim = sim
        # -> time resolution information
        self.T, self.t, self.dt = steps, np.arange(steps), 1 / (steps/ 24)
        self.time_range = pd.date_range(start=start_date, end=end_date + td(days=1), freq=resolution)[:-1]
        self.date_range = pd.date_range(start=start_date, end=end_date, freq="d")
        self.resolution = resolution
        # -> input parameters and time series for the optimization
        self.weather = pd.DataFrame(index=self.time_range)
        self.tariff = pd.DataFrame(index=self.time_range)
        self.grid_fee = pd.DataFrame(index=self.time_range)
        if profile_generator is None:
            self._demand_p = PowerProfile(demandP=1000, type=consumer_type, resolution=self.T)
            self._demand_q = HeatProfile(demandQ=10000)
        else:
            self._demand_p = profile_generator['power']
            self._demand_q = profile_generator['heat']

        self._pv_systems = {'systems': pv_systems, 'radiation': []}
        self.pv_capacity = sum([s.arrays[0].module_parameters["pdc0"] for s in pv_systems if s is not None])

        # -> random with set seed
        self.random = random
        # -> optimization output time series
        self._steps = len(self.time_range)

        self.strategy = strategy
        self.dispatcher = None

        self.finished, self.initial_plan = False, True
        self.next_request = self.time_range[0] - td(minutes=1)

        self._database = create_engine(database_uri)

        # -> dataframe to store simulation results
        self._data = pd.DataFrame(
            columns=[
                "consumer_id",
                "demand",
                "heat_hot_water",
                "COP_hot_water",
                "heat_space",
                "COP_space",
                "residual_demand",
                "generation",
                "residual_generation",
                "grid_fee",
                "car_demand",
                "planned_grid_consumption",
                "final_grid_consumption",
                "planned_grid_feed_in",
                "final_grid_feed_in",
                "planned_pv_consumption",
                "final_pv_consumption",
            ],
            index=self.time_range,
        )

        for column in self._data.columns:
            self._data[column] = np.zeros(self._steps)

        # -> set consumer id
        self._data.loc[self.time_range, "consumer_id"] = [consumer_id] * self._steps
        # -> set tariff data
        if tariff is None:
            tariff = pd.Series(data=30 * np.ones(self._steps), index=self.time_range)
        self._data.loc[tariff.index, "tariff"] = tariff.values.flatten()
        self.tariff = tariff.copy()

        # -> set grid fee data
        if grid_fee is None:
            grid_fee = pd.Series(data=5 * np.ones(self._steps), index=self.time_range)
        self._data.loc[grid_fee.index, "grid_fee"] = grid_fee.values.flatten()
        self.grid_fee = grid_fee.copy()

        # -> set weather data
        self.weather = weather.copy()

        # -> generate radiation series for each pv system
        for system in self._pv_systems['systems']:
            # -> irradiance unit [W/m²]
            rad = get_total_irradiance(
                solar_zenith=self.weather["zenith"],
                solar_azimuth=self.weather["azimuth"],
                dni=self.weather["dni"],
                ghi=self.weather["ghi"],
                dhi=self.weather["dhi"],
                surface_tilt=system.arrays[0].module_parameters["surface_tilt"],
                surface_azimuth=system.arrays[0].module_parameters["surface_azimuth"],
            )
            self._pv_systems['radiation'].append(rad)

        # -> calculate demand at each day
        demand_at_each_day = []
        for day in self.date_range:
            demand = self._demand_p.run_model(day)
            demand_at_each_day.append(resample_time_series(demand, self.T))
        demand_at_each_day = np.hstack(demand_at_each_day)
        self._data.loc[self.time_range, "demand"] = demand_at_each_day

        for day in self.date_range:
            time_range = pd.date_range(start=day, periods=self.T, freq=self.resolution)
            temperature = self.weather.loc[time_range, 'temp_air'].values - 273.15
            result = self._demand_q.run_model(temperature)
            self._data.loc[time_range, "heat_hot_water"] = result['hot_water']
            self._data.loc[time_range, "COP_hot_water"] = result['cop_hot_water']
            self._data.loc[time_range, "heat_space"] = result['space_heating']
            self._data.loc[time_range, "COP_space"] = result['cop_space_heating']

        # -> calculate generation at each day
        generation_at_each_day = np.zeros(self._steps)
        for system, rad in zip(self._pv_systems["systems"], self._pv_systems["radiation"]):
            peak_power = system.arrays[0].module_parameters["pdc0"]
            power = (rad["poa_global"] / 1e3 * peak_power)
            generation_at_each_day += resample_time_series(power.values, self.T)
        self._data.loc[self.time_range, "generation"] = generation_at_each_day

        # -> set residual time series
        residual_demand = self._data["demand"] - self._data["generation"]
        residual_demand[residual_demand < 0] = 0
        self._data.loc[self.time_range, "residual_demand"] = residual_demand
        residual_generation = self._data["generation"] - self._data["demand"]
        residual_generation[residual_generation < 0] = 0
        self._data.loc[self.time_range, "residual_generation"] = residual_generation

    def has_commit(self, d_time: datetime):
        if d_time > self.next_request:
            self.finished = False
            self.initial_plan = True
        return self.finished

    def simulate(self, d_time):
        for driver in self.drivers:
            if driver.car.type == "ev":
                # -> do charging
                driver.car.charge(d_time)
                # -> do driving
                driver.car.drive(d_time)

    def get_request(self, d_time: datetime) -> pd.Series:
        if self.dispatcher is None:
            self.finished = True
            self.next_request = d_time + td(days=1) - td(minutes=1)
        return pd.Series(dtype=float, index=[d_time], data=[0])

    def get(self, data_type: DataType, time_range=None, build_dataframe: bool = False):
        time_range = time_range or self.time_range
        result = self._data.loc[time_range, data_type.name]
        if build_dataframe:
            dataframe = pd.DataFrame({"power": result.values})
            dataframe.index = self.time_range
            dataframe["id_"] = str(self.id_)
            dataframe["node_id"] = self.grid_node
            dataframe = dataframe.rename_axis("t")
            return dataframe
        return result

    def get_result(self, time_range: pd.DatetimeIndex = None) -> pd.DataFrame:
        if time_range is None:
            time_range = self.time_range
        return self._data.loc[time_range]

    def save_ev_data(self, d_time: datetime):
        pass
