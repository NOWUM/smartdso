import logging
import uuid
from datetime import datetime
from datetime import timedelta as td

import numpy as np
import pandas as pd
from sqlalchemy import create_engine

from agents.utils import WeatherGenerator
from participants.basic import BasicParticipant, DataType

logger = logging.getLogger("smartdso.flexibility_provider")


# -> default prices EPEX-SPOT 2015
default_price = r"./agents/data/default_prices.csv"
TARIFF = pd.read_csv(default_price, index_col=0, parse_dates=True)
TARIFF = TARIFF / 10  # -> [€/MWh] in [ct/kWh]
CHARGES = {"others": 2.9, "taxes": 8.0}
for values in CHARGES.values():
    TARIFF += values


class FlexibilityProvider:
    def __init__(
        self,
        name: str,
        sim: int,
        start_date: datetime,
        end_date: datetime,
        database_uri: str,
        t: int = 96,
        resolution: str = '15min',
        tariff: str = 'flat',
        *args,
        **kwargs,
    ):

        # -> scenario name and iteration number
        self.name = name
        self.sim = sim
        # -> simulation time range and steps per day
        self.time_range = pd.date_range(start=start_date, end=end_date + td(days=1), freq=resolution)[-1]
        self.date_range = pd.date_range(start=start_date, end=end_date + td(days=1), freq="d")
        self.T = t
        # -> total clients
        self.clients: dict[uuid.UUID, BasicParticipant] = {}
        # -> weather generator
        self.weather_generator = WeatherGenerator()
        # -> database connection
        self._database = create_engine(database_uri)
        # -> set dynamic or flat enerhy price
        if tariff == 'flat':  # -> use median
            self.tariff[TARIFF.index, "tariff"] = TARIFF.values.mean()
        else:
            self.tariff = TARIFF

        # -> initialize list with all residential clients
        self.keys = []
        self._commits = {}
        for key, value in self.clients.items():
            if value.consumer_type == "household":
                self.keys.append(key)

    def register(self, id_: uuid.uuid1, client: BasicParticipant):
        self.clients[id_] = client
        if client.consumer_type == "household":
            self.keys.append(id_)
            self._commits[id_] = False

    def get_tariff(self, time_range: pd.DatetimeIndex = None):
        if time_range is None:
            time_range = self.time_range
        return self.tariff.loc[time_range]

    def get_commits(self) -> int:
        return sum([int(c) for c in self._commits.values()])

    def get_requests(self, d_time: datetime, random) -> (pd.Series, str):
        random.shuffle(self.keys)
        for id_ in self.keys:
            self._commits[id_] = self.clients[id_].has_commit()
            if not self._commits[id_]:
                request = self.clients[id_].get_request(d_time)
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

        total_demand, car_counter = np.zeros(self.T), 0

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
