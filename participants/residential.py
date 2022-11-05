import logging
from datetime import datetime
from datetime import timedelta as td

import numpy as np
import pandas as pd
from pvlib.pvsystem import PVSystem

from carLib.car import CarData, Car
from demLib.electric_profile import StandardLoadProfile
from participants.strategy.system_dispatch import EnergySystemDispatch
from participants.basic import BasicParticipant, DataType
from participants.resident import Resident

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
            steps: int = 96,
            resolution: str = "15min",
            consumer_id: str = "nowum",
            strategy: str = "MaxPvCap",
            sub_grid: int = -1,
            weather: pd.DataFrame = None,
            tariff: pd.Series = None,
            grid_fee: pd.Series = None,
            max_request: int = 5,
            *args,
            **kwargs,
    ):

        # -> initialize profile generator
        p_gen = StandardLoadProfile(demandP=demand_power, london_data=london_data,
                                    l_id=london_id, resolution=steps)
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
            profile_generator=p_gen,
            pv_systems=pv_systems,
            weather=weather,
            tariff=tariff,
            grid_fee=grid_fee,
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
        # self._maximal_benefit = self._y_data[-1]
        logger.info(f" -> set maximal iteration to {max_request}")
        self._max_requests = max_request

        self.dispatcher = EnergySystemDispatch(
            steps=self.T,
            resolution=self.resolution,
            strategy=strategy,
            benefit_function=self.b_fnc,
            generation=self.get(DataType.residual_generation),
            tariff=tariff,
            grid_fee=grid_fee,
            electric_vehicles=list(self.cars.values())
        )

    def get_request(self, d_time: datetime):
        if self._total_capacity > 0 and d_time > self._commit:
            if "MaxPv" in self.strategy:
                pass
                # self._optimize_photovoltaic_usage(d_time=d_time)
            elif "PlugIn" in self.strategy:
                pass
                # self._plan_without_photovoltaic(d_time=d_time)
            else:
                logger.error(f"invalid strategy {self.strategy}")
                raise Exception(f"invalid strategy {self.strategy}")

        elif self._total_capacity == 0:
            self._finished = True
            self._initial_plan = True

        return self._request

    def commit(self, price: pd.Series):
        tariff = self._data.loc[price.index, "tariff"].values.flatten()
        grid_fee = price.values.flatten()
        total_price = sum((tariff + grid_fee) * self._request.values) * self.dt
        final_tariff = self._data.loc[price.index, "tariff"] + price
        # print(final_tariff)
        if self._benefit_value > total_price or self._max_requests == 0:
            for key, car in self.cars.items():
                car.set_final_charging(self._car_power[key])
                car.set_final_tariff(final_tariff)
            self._commit = price.index.max()
            self._data.loc[price.index, "final_grid_consumption"] = self._request.loc[
                price.index
            ].copy()
            self._data.loc[:, "final_pv_consumption"] = self._data.loc[
                                                        :, "planned_pv_consumption"
                                                        ].copy()
            self._request = pd.Series(data=np.zeros(len(price)), index=price.index)
            self._data.loc[price.index, "grid_fee"] = price.values
            self._data = self._data.fillna(0)
            self._max_requests = 5
            if "MaxPv" in self.strategy:
                self._finished = True
            self._initial_plan = True
            return True
        else:
            if "MaxPv" in self.strategy:
                self._data.loc[price.index, "grid_fee"] = price.values
            else:
                self._commit += td(minutes=np.random.randint(low=1, high=3))
            self._max_requests -= 1

            return False
