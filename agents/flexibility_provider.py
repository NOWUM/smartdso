import logging
from tqdm import tqdm
from datetime import datetime
from datetime import timedelta as td

import numpy as np
import pandas as pd
from sqlalchemy import create_engine

from agents.utils import WeatherGenerator
from participants.basic import BasicParticipant
from carLib.car import CarData

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
        random: np.random.default_rng,
        database_uri: str,
        steps: int = 96,
        resolution: str = '15min',
        tariff: str = 'flat',
        *args,
        **kwargs,
    ):

        # -> scenario name and iteration number
        self.name = name
        self.sim = sim
        # -> simulation time range and steps per day
        self.time_range = pd.date_range(start=start_date, end=end_date + td(days=1), freq=resolution)[:-1]
        self.date_range = pd.date_range(start=start_date, end=end_date + td(days=1), freq="d")
        self.T = steps
        self.resolution = resolution
        # -> total clients
        self.clients: dict[str, BasicParticipant] = {}
        # -> weather generator
        self.weather_generator = WeatherGenerator()
        # -> database connection
        self._database = create_engine(database_uri)
        # -> set dynamic or flat energy price
        if tariff == 'flat':  # -> use median
            median = TARIFF.loc[:, 'price'].median()
            self.tariff = pd.Series(data=[median]*len(self.time_range), index=self.time_range, name='price')
        else:
            self.tariff = TARIFF
        self.tariff = self.tariff.resample(resolution).ffill()

        self.keys = []
        self._commits = {}

        self.random = random
        self.consumer_handler = None

    def register(self, id_: str, client: BasicParticipant):
        self.clients[id_] = client
        self.keys.append(id_)
        self._commits[id_] = False

    def get_tariff(self, time_range: pd.DatetimeIndex = None):
        if time_range is None:
            time_range = self.time_range
        return self.tariff.loc[time_range]

    def get_commits(self) -> int:
        return sum([int(c) for c in self._commits.values()])

    def get_requests(self, d_time: datetime) -> (pd.Series, str):
        self.random.shuffle(self.keys)
        for id_ in tqdm(self.keys):
            self.consumer_handler = self.clients[id_]
            commit = self.consumer_handler.has_commit(d_time)
            if not commit:
                request = self.consumer_handler.get_request(d_time)
                if sum(request.values) != 0:
                    yield request, self.consumer_handler.grid_node
                else:
                    self._commits[id_] = self.consumer_handler.finished

    def commit(self, price: pd.Series) -> bool:
        commit = self.consumer_handler.commit(price=price)
        if commit:
            id_ = self.consumer_handler.id_
            self._commits[id_] = True
        return commit

    def save_consumer_summary(self, d_time: datetime) -> None:

        time_range = pd.date_range(start=d_time, freq=self.resolution, periods=self.T)

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

        result["sub_id"] = -1

        for id_, client in self.clients.items():
            # -> reset parameter for optimization for the next day
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
                result["availability"].loc[time_range] += 1 - car.get(CarData.usage, time_range)

            result.loc[time_range, 'sub_id'] = client.sub_grid

        result["availability"] /= car_counter
        result["grid_fee"] /= total_demand

        result["scenario"] = self.name
        result["iteration"] = self.sim

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
