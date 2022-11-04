import os
from datetime import datetime
from datetime import timedelta as td
from enum import Enum

import numpy as np
import pandas as pd
from pvlib.irradiance import get_total_irradiance
from sqlalchemy import create_engine

from carLib.car import Car, CarData
from demLib.electric_profile import StandardLoadProfile
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
        steps: int = 96,
        resolution: str = "15min",
        grid_node: str = None,
        start_date: datetime = None,
        end_date: datetime = None,
        consumer_type: str = "household",
        consumer_id: str = "nowum",
        strategy: str = "optimized",
        profile_generator: StandardLoadProfile = None,
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
        self._database = create_engine(database_uri)
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

        # -> time resolution information
        self.T, self.t, self.dt = steps, np.arange(steps), 1 / (steps/ 24)
        self.time_range = pd.date_range(start=start_date, end=end_date + td(days=1), freq=resolution)[:-1]
        self.date_range = pd.date_range(start=start_date, end=end_date, freq="d")
        # -> input parameters and time series for the optimization
        self.weather = pd.DataFrame(index=self.time_range)
        self.tariff = pd.DataFrame(index=self.time_range)
        self.grid_fee = pd.DataFrame(index=self.time_range)
        if profile_generator is None:
            self._demand_p = StandardLoadProfile(demandP=1000, type=consumer_type, resolution=self.T)
        else:
            self._demand_p = profile_generator
        self._pv_systems = {'systems': pv_systems, 'radiation': []}
        self.pv_capacity = sum([s.arrays[0].module_parameters["pdc0"] for s in pv_systems if s is not None])

        # -> random with set seed
        self.random = random
        # -> optimization output time series
        self._steps = len(self.time_range)

        self._request = pd.Series(dtype=float)
        self.strategy = strategy

        self._finished, self._initial_plan = False, True
        self._commit = self.time_range[0] - td(minutes=1)

        # -> dataframe to store simulation results
        self._data = pd.DataFrame(
            columns=[
                "consumer_id",
                "demand",
                "residual_demand",
                "generation",
                "residual_generation",
                "grid_fee",
                "car_demand",
                "planned_grid_consumption",
                "final_grid_consumption",
                "planned_pv_consumption",
                "final_pv_consumption",
            ],
            index=self.time_range,
        )

        for column in self._data.columns:
            self._data[column] = np.zeros(len(self.time_range))

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
        self._data.loc[grid_fee.index, "tariff"] = grid_fee.values.flatten()
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

    def has_commit(self) -> bool:
        return self._finished

    def reset_commit(self) -> None:
        self._finished = False
        if self.strategy == "optimized":
            self._initial_plan = True

    def simulate(self, d_time):
        for person in [p for p in self.persons if p.car.type == "ev"]:
            person.car.charge(d_time)   # -> do charging
            person.car.drive(d_time)    # -> do driving

    def get_request(self, d_time: datetime) -> pd.Series:
        if d_time > self._commit:
            self._commit = d_time + td(days=1)
            self._finished = True
            self._initial_plan = True
        return pd.Series(dtype=float, index=[d_time], data=[0])

    def get(self, data_type: DataType, time_range=None):
        time_range = time_range or self.time_range
        return self._data.loc[time_range, data_type.name]

    def get_result(self, time_range: pd.DatetimeIndex = None) -> pd.DataFrame:
        time_range = time_range or self.time_range
        return self._data.loc[time_range]

    def get_initial_power(self, data_type: DataType):
        data = self.get(data_type=data_type)

        dataframe = pd.DataFrame({"power": data.values})
        dataframe.index = self.time_range
        dataframe["id_"] = str(self.id_)
        dataframe["node_id"] = self.grid_node
        dataframe = dataframe.rename_axis("t")

        return dataframe