import logging
from datetime import datetime
from datetime import timedelta as td

import numpy as np
import pandas as pd
from pvlib.pvsystem import PVSystem

from carLib.car import CarData, Car
from demLib.electric_profile import StandardLoadProfile as PowerProfile
from demLib.heat_profile import StandardLoadProfile as HeatProfile
from participants.strategy.system_dispatch import EnergySystemDispatch
from participants.basic import BasicParticipant, DataType
from participants.resident import Resident
from participants.utils import HeatStorage

# example to Piecewise:
# http://yetanothermathprogrammingconsultant.blogspot.com/2019/02/piecewise-linear-functions-and.html
# http://yetanothermathprogrammingconsultant.blogspot.com/2015/10/piecewise-linear-functions-in-mip-models.html


# -> set logging options
LOG_LEVEL = "INFO"
logger = logging.getLogger("smartdso.residential")
logger.setLevel(LOG_LEVEL)

benefit_functions = r"./participants/strategy/benefit_function.csv"
# -> benefit functions from survey
BENEFIT_FUNCTIONS = pd.read_csv(benefit_functions, index_col=0)
# -> adjust to actually market level
BENEFIT_FUNCTIONS += 0.10
BENEFIT_FUNCTIONS *= 100
CUM_PROB = np.cumsum([float(x) for x in BENEFIT_FUNCTIONS.columns])

KEY = 0


def key_generator(sub_grid: int, pv_capacity: float, ev_capacity: float):
    global KEY
    KEY += 1
    return f"S{sub_grid}C{KEY}_{pv_capacity}_{ev_capacity}"


class HouseholdModel(BasicParticipant):
    def __init__(
            self,
            residents: int,
            demand_power: float,
            demand_heat: float,
            database_uri: str,
            random: np.random.default_rng,
            london_data: bool = False,
            london_id: str = "MAC002957",
            ev_ratio: float = 0.5,
            pv_systems: list = None,
            grid_node: str = None,
            start_date: datetime = datetime(2022, 1, 1),
            end_date: datetime = datetime(2022, 1, 2),
            name: str = "testing",
            sim: int = 0,
            steps: int = 96,
            resolution: str = "15min",
            consumer_id: str = "nowum",
            strategy: str = "MaxPvCap",
            sub_grid: int = -1,
            weather: pd.DataFrame = None,
            tariff: pd.Series = None,
            grid_fee: pd.Series = None,
            max_request: int = 4,
            *args,
            **kwargs,
    ):

        # -> initialize profile generator
        p_gen = PowerProfile(demandP=demand_power, london_data=london_data, l_id=london_id, resolution=steps)
        q_gen = HeatProfile(demandQ=demand_heat)

        # -> initialize pv systems
        pv_systems = [PVSystem(module_parameters=system) for system in pv_systems]

        super().__init__(
            steps=steps,
            resolution=resolution,
            consumer_id=consumer_id,
            sub_grid=sub_grid,
            grid_node=grid_node,
            residents=residents,
            start_date=start_date,
            end_date=end_date,
            database_uri=database_uri,
            consumer_type="household",
            strategy=strategy,
            random=random,
            profile_generator={'power': p_gen, 'heat': q_gen},
            pv_systems=pv_systems,
            weather=weather,
            tariff=tariff,
            grid_fee=grid_fee,
            name=name,
            sim=sim
        )

        # -> generate driver and cars
        logger.info(" -> generate drivers and cars")
        self.drivers: list[Resident] = []
        self.cars: dict[str, Car] = {}
        self._car_power: dict[str, pd.Series] = {}
        self._total_capacity = 0
        for _ in range(min(residents, 2)):
            driver = Resident(
                ev_ratio=ev_ratio,
                charging_limit="required",
                start_date=start_date,
                end_time=end_date,
                T=self.T,
                random=random,
            )
            self.drivers.append(driver)
            if driver.car.type == "ev":
                key = key_generator(self.sub_grid, self.pv_capacity, driver.car.capacity)
                self.cars[key] = driver.car
                self._car_power[key] = pd.Series(dtype=float)
                self._total_capacity += driver.car.capacity

        car_demand_at_each_day = np.zeros(self._steps)
        for car in self.cars.values():
            car_demand = car.get(CarData.demand).values
            car_demand_at_each_day += car_demand
        self._data.loc[self.time_range, "car_demand"] = car_demand_at_each_day

        logger.info(f" -> generated {len(self.cars)} EV with corresponding drivers")

        # -> setting strategy options
        logger.info(" -> setting strategy options")
        if "Cap" in strategy:
            self.price_limit = 45
            self.b_fnc = pd.Series(data=[self.price_limit] * 20, index=[*range(5, 105, 5)])
            logger.info(f" -> set price limit to {self.price_limit} ct/kWh")
        elif "Inf" in strategy:
            self.price_limit = 9_999
            self.b_fnc = pd.Series(data=[self.price_limit] * 20, index=[*range(5, 105, 5)])
            logger.info(f" -> no price limit is set")
        else:
            col = np.argwhere(np.random.uniform() * 100 > CUM_PROB).flatten()
            col = col[-1] if len(col) > 0 else 0
            self.b_fnc = BENEFIT_FUNCTIONS.iloc[:, col]
            logger.info(f" -> set benefit function {col} with mean price limit of {self.b_fnc.values.mean()} ct/kWh")

        logger.info(f" -> set maximal iteration to {max_request}")
        self._max_requests = [max_request, max_request]

        self.heat_storages = {'space': HeatStorage(volume=400, d_theta=35),
                              'hot_water': HeatStorage(volume=50 * residents, d_theta=50)}

        self.dispatcher = EnergySystemDispatch(
            steps=self.T,
            resolution=self.resolution,
            strategy=strategy,
            benefit_function=self.b_fnc,
            generation=self.get(DataType.residual_generation),
            tariff=tariff,
            grid_fee=grid_fee,
            electric_vehicles=self.cars,
            analyse_ev=True if self._total_capacity > 0 else False,
            heat_demand=self._data.loc[:, ['heat_hot_water', 'COP_hot_water',
                                           'heat_space', 'COP_space']],
            heat_storages=self.heat_storages,
            heat_pump=self._demand_q.max_demand
        )

    def get_request(self, d_time: datetime):

        grid = super().get_request(d_time)

        if self.dispatcher is not None and d_time > self.next_request:
            if "optimize" in self.strategy:
                self.dispatcher.get_optimal_solution(d_time)
            elif "heuristic" in self.strategy:
                self.dispatcher.get_heuristic_solution(d_time)
            else:
                logger.error(f"invalid strategy {self.strategy}")
                raise Exception(f"invalid strategy {self.strategy}")

            # -> get grid consumption
            grid = self.dispatcher.request
            if self.initial_plan:
                self.initial_plan = False
                initial_grid = self.dispatcher.grid_out
                self._data.loc[initial_grid.index, "planned_grid_consumption"] = initial_grid.copy()
                initial_grid = self.dispatcher.grid_in
                self._data.loc[initial_grid.index, "planned_grid_feed_in"] = initial_grid.copy()
                initial_pv = self.dispatcher.pv_usage
                self._data.loc[initial_pv.index, "planned_pv_consumption"] = initial_pv.copy()

            if all(grid.values == 0):
                index = pd.date_range(start=d_time, periods=self.T, freq=self.resolution)
                price = pd.Series(index=index, data=np.zeros(self.T))
                self._max_requests[0] = 0
                self.commit(price=price)

        elif self.dispatcher is None:
            self.finished = True

        return grid

    def commit(self, price: pd.Series):
        # -> calculate total charging costs
        power = self.dispatcher.request.values
        tariff = self._data.loc[price.index, "tariff"].values.flatten()
        grid_fee = price.values.flatten()
        total_price = sum((tariff + grid_fee) * np.abs(power)) * self.dt
        # -> get benefit value
        benefit = self.dispatcher.ev_benefit + self.dispatcher.hp_benefit + self.dispatcher.pv_benefit
        # -> compare benefit and costs
        if benefit > total_price or self._max_requests[0] == 0:
            # -> calculate final tariff time series
            final_tariff = self._data.loc[price.index, "tariff"] + price
            time_range = slice(price.index[0], price.index[-1])
            # -> set charging power for each car
            for key, car in self.cars.items():
                charging = car.get(CarData.planned_charge, time_range)
                car.set_final_charging(charging)
                car.set_final_tariff(final_tariff)
            # -> set volume for each storage type
            for key, storage in self.heat_storages.items():
                usage = storage.get_planned_usage()
                storage.set_planned_usage(usage)
            # -> set next request time
            self.next_request = price.index.max()
            # -> set final grid consumption
            final_grid = self.dispatcher.grid_out.loc[price.index]
            self._data.loc[price.index, "final_grid_consumption"] = final_grid.copy()
            # -> set final grid feed in
            final_grid = self.dispatcher.grid_in.loc[price.index]
            self._data.loc[price.index, "final_grid_feed_in"] = final_grid.copy()
            # -> set final pv consumption
            final_pv = self.dispatcher.pv_usage.loc[price.index]
            self._data.loc[:, "final_pv_consumption"] = final_pv.copy()
            # -> set final grid fee
            self._data.loc[price.index, "grid_fee"] = price.values
            # -> fill na with any went wrong
            self._data = self._data.fillna(0)
            # -> reset request counter
            self._max_requests[0] = self._max_requests[1]

            self.finished = True

            return True
        else:
            if "optimize" in self.strategy:
                self._data.loc[price.index, "grid_fee"] = price.values
                self.dispatcher.grid_fee.loc[price.index] = price.values
            else:
                self.next_request += td(minutes=np.random.randint(low=1, high=3))

            self._max_requests[0] -= 1

            return False

    def save_ev_data(self, d_time: datetime):
        time_range = pd.date_range(start=d_time, freq=self.resolution, periods=self.T)

        columns = {
            "soc": "soc",
            "usage": "usage",
            "planned_charge": "initial_charging",
            "final_charge": "final_charging",
            "demand": "demand",
            "distance": "distance",
            "tariff": "tariff"
        }

        index_columns = ["time", "scenario", "iteration", "sub_id", "id_"]

        for key, car in self.cars.items():
            data = car.get_result(time_range)
            data = data.loc[time_range, list(columns.keys())]
            data = data.rename(columns=columns)
            data["scenario"] = self.name
            data['iteration'] = self.sim
            data["sub_id"] = self.sub_grid
            data["id_"] = key
            data["pv"] = self.get(DataType.final_pv_consumption)
            data["pv_available"] = self.get(DataType.residual_generation)

            data.index.name = "time"
            data = data.reset_index()
            data = data.set_index(index_columns)

            try:
                data.to_sql(
                    name="electric_vehicle",
                    con=self._database,
                    if_exists="append",
                    method="multi",
                )
            except Exception as e:
                logger.warning(f"server closed the connection {repr(e)}")


if __name__ == "__main__":

    from matplotlib import pyplot as plt

    import numpy as np
    import pandas as pd
    from datetime import timedelta as td

    from agents.flexibility_provider import FlexibilityProvider
    from agents.capacity_provider import CapacityProvider
    from agents.utils import WeatherGenerator
    from config import SimulationConfig as Config

    config_dict = Config().get_config_dict()

    time_range = pd.date_range(start=Config.START_DATE, end=Config.END_DATE + td(days=1), freq=Config.RESOLUTION)[:-1]
    date_range = pd.date_range(start=Config.START_DATE, end=Config.END_DATE + td(days=1), freq="d")
    random = np.random.default_rng(Config.SEED)

    WeatherGenerator = WeatherGenerator()
    # -> get weather data
    weather_at_each_day = []
    for day in date_range:
        weather_at_each_day.append(WeatherGenerator.get_weather(day))
    weather_at_each_day = pd.concat(weather_at_each_day)
    # -> resample to quarter hour values
    weather_at_each_day = weather_at_each_day.resample("15min").ffill()
    # -> remove not used time steps
    idx = weather_at_each_day.index.isin(list(time_range))
    weather_at_each_day = weather_at_each_day.loc[idx]

    # -> start capacity provider
    CapProvider = CapacityProvider(**config_dict)
    grid_fee = CapProvider.get_grid_fee(time_range=time_range)
    logging.getLogger("smartdso.capacity_provider").setLevel("DEBUG")

    # -> start flexibility provider
    FlexProvider = FlexibilityProvider(random=random, **config_dict)
    tariff = FlexProvider.get_tariff(time_range=time_range)
    logging.getLogger("smartdso.flexibility_provider").setLevel("DEBUG")
    config_dict['tariff'] = tariff
    # -> start consumer agents
    consumer_path = rf"./gridLib/data/export/{Config.GRID_DATA}/consumers.csv"
    consumers = pd.read_csv(consumer_path, index_col=0)
    # -> all residential consumers
    residential_consumers = consumers.loc[consumers["profile"] == "H0"]
    residential_consumers = residential_consumers.fillna('[]')
    consumer = residential_consumers.iloc[0, :]

    r = HouseholdModel(
            demand_power=consumer['demand_power'],
            demand_heat=consumer['demand_heat'],
            consumer_id='test',
            grid_node=consumer["bus0"],
            residents=int(max(consumer["demand_power"] / 1500, 1)),
            london_id=consumer["london_data"],
            pv_systems=eval(consumer["pv"]),
            consumer_type="household",
            random=np.random.default_rng(2022),
            weather=weather_at_each_day,
            grid_fee=grid_fee,
            **config_dict
        )

    # r.dispatcher.get_optimal_solution(date_range[0])
    # ax = r.dispatcher.request.plot()
    # r.dispatcher.tariff.iloc[:96].plot(ax=ax)
    # vol_space = pd.Series(data=r.heat_storages['space'].get_planned_usage(),
    #                       index=r.dispatcher.tariff.iloc[:96].index)
    # vol_space.plot(ax=ax)
    # vol_water = pd.Series(data=r.heat_storages['hot_water'].get_planned_usage(),
    #                       index=r.dispatcher.tariff.iloc[:96].index)
    # vol_water.plot(ax=ax)
    # plt.show()
    # r.dispatcher.get_optimal_solution(date_range[0])
    # r.dispatcher.grid_fee.iloc[:50] = 150
    # ax = r.dispatcher.request.plot()
    # r.dispatcher.tariff.iloc[:96].plot(ax=ax)
    # ax.plot(r.dispatcher.request.index, r.heat_storages['space'].get_planned_usage())
    # plt.show()


    # model = r.dispatcher.m


