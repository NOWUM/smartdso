import logging
import os
import uuid
from datetime import datetime
from datetime import timedelta as td

import numpy as np
import pandas as pd
from sqlalchemy import create_engine

from agents.utils import WeatherGenerator
from participants.basic import BasicParticipant, DataType

if "profile" not in consumers.columns:
    consumers["profile"] = "H0"
if "pv" not in consumers.columns:
    consumers["pv"] = 0
if "jeb" not in consumers.columns:
    consumers["jeb"] = 4500
if "london_data" not in consumers.columns:
    consumers["london_data"] = "MAC001844"

from config import DATABASE_URI, RESOLUTION

logger = logging.getLogger("smartdso.flexibilityprovider")


class FlexibilityProvider:
    def __init__(
        self,
        scenario: str,
        iteration: int,
        clients: dict[uuid.UUID, BasicParticipant],
        start_date: datetime,
        end_date: datetime,
        sub_grid: int,
        database_uri: str,
        ev_ratio: float = 0.5,
        london_data: bool = False,
        pv_ratio: float = 0.3,
        T: int = 1440,
        number_consumers: int = 0,
        strategy: str = "MaxPvCap",
        *args,
        **kwargs,
    ):

        # -> scenario name and iteration number
        self.scenario = scenario
        self.iteration = iteration
        self.strategy = strategy
        # -> total clients
        self.clients: dict[uuid.UUID, BasicParticipant] = clients
        # -> weather generator
        self.weather_generator = WeatherGenerator()
        # -> time range
        self.time_range = pd.date_range(
            start=start_date, end=end_date + td(days=1), freq=RESOLUTION[T]
        )[:-1]

        self._database = create_engine(database_uri)

        self.T = T

        self.random = np.random.default_rng(SEED)

        self.sub_grid = sub_grid

        self.keys = [
            key
            for key, value in self.clients.items()
            if value.consumer_type == "household"
        ]
        self._commits = {key: False for key in self.keys}

    def initialize_time_series(self) -> (pd.DataFrame, pd.DataFrame):
        def build_dataframe(data, id_):
            dataframe = pd.DataFrame({"power": data.values})
            dataframe.index = self.time_range
            dataframe["id_"] = str(id_)
            dataframe["node_id"] = client.grid_node
            dataframe = dataframe.rename_axis("t")
            return dataframe

        weather = pd.concat(
            [
                self.weather_generator.get_weather(date=date)
                for date in pd.date_range(
                    start=self.time_range[0],
                    end=self.time_range[-1] + td(days=1),
                    freq="d",
                )
            ]
        )
        weather = weather.resample("15min").ffill()
        weather = weather.loc[weather.index.isin(self.time_range)]

        demand_, generation_ = [], []
        for id_, client in self.clients.items():
            client.set_parameter(weather=weather.copy())
            client.initial_time_series()
            demand = client.get(DataType.residual_demand)
            demand_.append(build_dataframe(demand, id_))
            generation = client.get(DataType.residual_generation)
            generation_.append(build_dataframe(generation, id_))

        return pd.concat(demand_), pd.concat(generation_)

    def get_commits(self) -> int:
        return sum([int(c) for c in self._commits.values()])

    def get_requests(self, d_time: datetime) -> (pd.Series, str):
        self.random.shuffle(self.keys)
        for id_ in self.keys:
            self._commits[id_] = self.clients[id_].has_commit()
            if not self._commits[id_]:
                request = self.clients[id_].get_request(d_time, strategy=self.strategy)
                if sum(request.values) > 0:
                    yield request, self.clients[id_].grid_node, id_

    def simulate(self, d_time: datetime) -> None:
        capacity, empty, pool = 0, 0, 0
        for participant in self.clients.values():
            participant.simulate(d_time)
            for person in [p for p in participant.persons if p.car.type == "ev"]:
                capacity += person.car.soc * person.car.capacity
                empty += int(person.car.empty)
                pool += person.car.virtual_source

    def commit(self, price: pd.Series, consumer_id: uuid.UUID) -> bool:
        commit_ = self.clients[consumer_id].commit(price=price)
        if commit_:
            self._commits[consumer_id] = self.clients[consumer_id].has_commit()
        return commit_

    def _save_summary(self, d_time: datetime) -> None:

        time_range = pd.date_range(
            start=d_time, freq=RESOLUTION[self.T], periods=self.T
        )

        result = pd.DataFrame(
            index=time_range,
            columns=[
                "initial_grid",
                "final_grid",
                "final_pv",
                "residential_demand",
                "car_demand",
                "residual_generation",
                "availability",
                "grid_fee",
            ],
        )
        result = result.fillna(0)

        total_demand, car_counter = np.zeros(96), 0

        for id_, client in self.clients.items():
            # -> reset parameter for optimization for the next day
            client.reset_commit()
            self._commits[id_] = False
            # -> get results for current day
            data = client.get_result(time_range)
            # -> collect results
            result.loc[time_range, "initial_grid"] += data.loc[
                time_range, "planned_grid_consumption"
            ]
            result.loc[time_range, "final_grid"] += data.loc[
                time_range, "final_grid_consumption"
            ]
            result.loc[time_range, "final_pv"] += data.loc[
                time_range, "final_pv_consumption"
            ]
            result.loc[time_range, "car_demand"] += data.loc[time_range, "car_demand"]
            result.loc[time_range, "residential_demand"] += data.loc[
                time_range, "demand"
            ]
            result.loc[time_range, "residual_generation"] += data.loc[
                time_range, "residual_generation"
            ]
            grid_fee = data.loc[time_range, "grid_fee"] * (
                data.loc[time_range, "final_grid_consumption"] / 4
            )
            result.loc[time_range, "grid_fee"] += grid_fee
            total_demand += data.loc[time_range, "final_grid_consumption"] / 4

            for key, car in client.cars.items():
                car_counter += 1
                result["availability"].loc[time_range] += 1 - car.get(
                    CarDataType.usage, time_range
                )

        result["availability"] /= car_counter
        result["grid_fee"] /= total_demand

        result["scenario"] = self.scenario
        result["iteration"] = self.iteration
        result["sub_id"] = self.sub_grid
        result = result.fillna(value=0)
        result.index.name = "time"
        result = result.reset_index()
        result = result.set_index(["time", "scenario", "iteration", "sub_id"])

        try:
            result.to_sql(
                name="charging_summary", con=self._database, if_exists="append"
            )
            logger.info("-> saved data in table: charging_summary")
            logger.info(
                f' -> sum pv: {round(result["final_pv"].sum(),1) / 4}, '
                f'sum grid: {round(result["final_grid"].sum(),1) / 4}, '
            )
        except Exception as e:
            logger.warning(f"server closed the connection {repr(e)}")

    def _save_electric_vehicles(self, d_time: datetime):
        time_range = pd.date_range(
            start=d_time, freq=RESOLUTION[self.T], periods=self.T
        )

        for id_, client in self.clients.items():
            for key, car in client.cars.items():
                data = car.get_result(time_range)
                data = data.loc[
                    :,
                    [
                        "soc",
                        "usage",
                        "planned_charge",
                        "final_charge",
                        "demand",
                        "distance",
                        "tariff",
                    ],
                ]
                data.columns = [
                    "soc",
                    "usage",
                    "initial_charging",
                    "final_charging",
                    "demand",
                    "distance",
                    "tariff",
                ]
                data["scenario"] = self.scenario
                data["iteration"] = self.iteration
                data["sub_id"] = self.sub_grid
                data["id_"] = key
                data["pv"] = client.get(DataType.final_pv_consumption)
                data["pv_available"] = client.get(DataType.residual_generation)

                data.index.name = "time"
                data = data.reset_index()
                data = data.set_index(
                    ["time", "scenario", "iteration", "sub_id", "id_"]
                )

                try:
                    data.to_sql(
                        name="electric_vehicle",
                        con=self._database,
                        if_exists="append",
                        method="multi",
                    )
                except Exception as e:
                    logger.warning(f"server closed the connection {repr(e)}")

    def save_results(self, d_time: datetime) -> None:
        self._save_summary(d_time)
        if self.iteration == 0:
            self._save_electric_vehicles(d_time)
