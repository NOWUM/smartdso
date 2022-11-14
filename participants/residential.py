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
            max_request: int = 5,
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

        self.heat_storages = {'space': HeatStorage(volume=500, d_theta=15),
                              'hot_water': HeatStorage(volume=500, d_theta=25)}

        self.dispatcher = None
        if self._total_capacity > 0:
            self.dispatcher = EnergySystemDispatch(
                steps=self.T,
                resolution=self.resolution,
                strategy=strategy,
                benefit_function=self.b_fnc,
                generation=self.get(DataType.residual_generation),
                tariff=tariff,
                grid_fee=grid_fee,
                electric_vehicles=self.cars,
                heat_demand=self._data.loc[:, ['heat_hot_water', 'COP_hot_water',
                                               'heat_space', 'COP_space']],
                heat_storages=self.heat_storages
            )

    def get_request(self, d_time: datetime):

        grid = super().get_request(d_time)

        if self._total_capacity > 0 and d_time > self.next_request:
            if "optimize" in self.strategy:
                self.dispatcher.get_optimal_solution(d_time)
            elif "heuristic" in self.strategy:
                self.dispatcher.get_heuristic_solution(d_time)
            else:
                logger.error(f"invalid strategy {self.strategy}")
                raise Exception(f"invalid strategy {self.strategy}")

            # -> get grid consumption
            grid = self.dispatcher.request
            # -> get pv consumption
            pv = self.dispatcher.pv_charge
            if self.initial_plan:
                self.initial_plan = False
                self._data.loc[grid.index, "planned_grid_consumption"] = grid.copy()
                self._data.loc[pv.index, "planned_pv_consumption"] = pv.copy()

            if grid.sum() == 0:
                index = pd.date_range(start=d_time, periods=self.T, freq=self.resolution)
                price = pd.Series(index=index, data=np.zeros(self.T))
                self.commit(price=price)

        elif self._total_capacity == 0:
            self.finished = True

        return grid

    def commit(self, price: pd.Series):
        # -> calculate total charging costs
        power = self.dispatcher.request.values
        tariff = self._data.loc[price.index, "tariff"].values.flatten()
        grid_fee = price.values.flatten()
        total_price = sum((tariff + grid_fee) * power) * self.dt
        # -> get benefit value
        benefit = self.dispatcher.benefit
        # -> compare benefit and costs
        if benefit > total_price or self._max_requests[0] == 0:
            # -> calculate final tariff time series
            final_tariff = self._data.loc[price.index, "tariff"] + price
            # -> set charging power for each car
            for key, car in self.cars.items():
                car.set_final_charging(self._car_power[key])
                car.set_final_tariff(final_tariff)
            # -> set next request time
            self.next_request = price.index.max()
            # -> set final grid consumption
            final_grid = self.dispatcher.request.loc[price.index]
            self._data.loc[price.index, "final_grid_consumption"] = final_grid.copy()
            # -> set final pv consumption
            final_pv = self.dispatcher.pv_charge.loc[price.index]
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
