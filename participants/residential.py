import logging
import os
from datetime import datetime
from datetime import timedelta as td
from math import exp

import numpy as np
import pandas as pd
from pvlib.pvsystem import PVSystem
from pyomo.environ import (
    Binary,
    ConcreteModel,
    Constraint,
    ConstraintList,
    Objective,
    Piecewise,
    Reals,
    SolverFactory,
    Var,
    maximize,
    minimize,
    quicksum,
    value,
)

from carLib.car import CarData
from config import DATABASE_URI, RESOLUTION, key_generator
from demLib.electric_profile import StandardLoadProfile
from participants.basic import BasicParticipant
from participants.resident import Resident

# example to Piecewise:
# http://yetanothermathprogrammingconsultant.blogspot.com/2019/02/piecewise-linear-functions-and.html
# http://yetanothermathprogrammingconsultant.blogspot.com/2015/10/piecewise-linear-functions-in-mip-models.html


# -> set logging options
LOG_LEVEL = "INFO"
logger = logging.getLogger("smartdso.residential")
logger.setLevel(LOG_LEVEL)

# -> benefit functions from survey
BENEFIT_FUNCTIONS = pd.read_csv(
    r"./participants/data/benefit_function.csv", index_col=0
)
BENEFIT_FUNCTIONS += 0.10
BENEFIT_FUNCTIONS *= 100
CUM_PROB = np.cumsum([float(x) for x in BENEFIT_FUNCTIONS.columns])
# -> default prices EPEX-SPOT 2015
TARIFF = pd.read_csv(
    r"./participants/data/2022_prices.csv", index_col=0, parse_dates=True
)
TARIFF = TARIFF / 10  # -> [€/MWh] in [ct/kWh]
CHARGES = {"others": 2.9, "taxes": 8.0}
for values in CHARGES.values():
    TARIFF += values


class HouseholdModel(BasicParticipant):
    def __init__(
        self,
        residents: int,
        demandP: float,
        random: np.random.default_rng,
        london_data: bool = True,
        l_id: str = "MAC002957",
        ev_ratio: float = 0.5,
        pv_systems: list = None,
        grid_node: str = None,
        start_date: datetime = datetime(2022, 1, 1),
        end_date: datetime = datetime(2022, 1, 2),
        tariff: pd.DataFrame = TARIFF,
        T: int = 1440,
        database_uri: str = DATABASE_URI,
        consumer_id: str = "nowum",
        strategy: str = "MaxPvCap",
        scenario: str = "Flat",
        *args,
        **kwargs,
    ):

        super().__init__(
            T=T,
            grid_node=grid_node,
            start_date=start_date,
            end_date=end_date,
            database_uri=database_uri,
            consumer_type="household",
            strategy=strategy,
            random=random,
        )

        # -> initialize profile generator
        self._profile_generator = StandardLoadProfile(
            demandP=demandP, london_data=london_data, l_id=l_id, resolution=self.T
        )
        # -> create residents with cars
        self.persons = [
            Resident(
                ev_ratio=ev_ratio,
                charging_limit="required",
                start_date=start_date,
                end_time=end_date,
                T=self.T,
                random=random,
            )
            for _ in range(min(2, residents))
        ]

        if "PlugInInf" in strategy:
            self.price_limit = np.inf
        else:
            self.price_limit = 45

        self._pv_systems = [PVSystem(module_parameters=system) for system in pv_systems]
        pv_capacity = sum([s["pdc0"] for s in pv_systems])

        self.cars = {
            key_generator(pv_capacity, person.car.capacity): person.car
            for person in self.persons
            if person.car.type == "ev"
        }

        if len(self.cars) > 0:
            self._total_capacity = sum([c.capacity for c in self.cars.values()])
            self._car_power = {c: pd.Series(dtype=float) for c in self.cars.keys()}
        else:
            self._total_capacity = 0

        col = np.argwhere(np.random.uniform() * 100 > CUM_PROB).flatten()
        col = col[-1] if len(col) > 0 else 0
        self.b_fnc = BENEFIT_FUNCTIONS.iloc[:, col]

        self._benefit_value = 0
        self._x_data = list(self.b_fnc.index / 100 * self._total_capacity)
        self._y_data = list(np.cumsum(self.b_fnc.values * self._total_capacity * 0.05))

        self._max_requests = 5
        self.finished = False

        self._model = ConcreteModel()
        self._solver_type = "glpk"
        self._solver = SolverFactory(self._solver_type)

        tariff.index = pd.date_range(
            start=datetime(start_date.year, 1, 1), freq="h", periods=len(tariff)
        )
        tariff = tariff.resample(RESOLUTION[self.T]).ffill().loc[self.time_range]
        self._data.loc[tariff.index, "tariff"] = tariff.values.flatten()
        if "Flat" in scenario:  # -> use median
            self._data.loc[tariff.index, "tariff"] = tariff.values.mean()
        self._data.loc[self.time_range, "grid_fee"] = self.random.normal(
            2.6, 1e-3, self._steps
        )

        self._data.loc[self.time_range, "pv_capacity"] = pv_capacity * np.ones(
            self._steps
        )
        self._data.loc[self.time_range, "consumer_id"] = [consumer_id] * self._steps
        self._data.loc[
            self.time_range, "car_capacity"
        ] = self._total_capacity * np.ones(self._steps)

    def _optimize_photovoltaic_usage(
        self, d_time: datetime, strategy: str = "required"
    ):

        capacity = 0
        for car in self.cars.values():
            capacity += car.get_current_capacity()

        # -> get residual generation and determine possible opt. time steps
        generation = self._data.loc[d_time:, "residual_generation"].values
        steps = range(min(self.T, len(generation)))
        demand = {
            key: car.get(CarData.demand, slice(d_time, None)).values[steps]
            for key, car in self.cars.items()
        }
        # -> get prices
        tariff = self._data.loc[d_time:, "tariff"].values.flatten()[steps]
        grid_fee = self._data.loc[d_time:, "grid_fee"].values.flatten()[steps]
        prices = tariff + grid_fee
        if self._max_requests == 0:
            prices += 1e6

        # -> clear model
        self._model.clear()
        # -> declare variables
        self._model.power = Var(self.cars.keys(), steps, within=Reals, bounds=(0, None))
        self._model.grid = Var(steps, within=Reals, bounds=(0, None))
        self._model.pv = Var(steps, within=Reals, bounds=(0, None))
        self._model.capacity = Var(within=Reals, bounds=(0, self._total_capacity))
        self._model.volume = Var(self.cars.keys(), steps, within=Reals)
        self._model.benefit = Var(within=Reals, bounds=(0, None))

        if "Soc" in self._used_strategy:
            if self._solver_type == "glpk":

                segments = dict(low=[], up=[], coeff=[], low_=[])
                for i in range(0, len(self._y_data) - 1):
                    segments["low"].append(self._x_data[i])
                    segments["up"].append(self._x_data[i + 1])
                    dy = self._y_data[i + 1] - self._y_data[i]
                    dx = self._x_data[i + 1] - self._x_data[i]
                    segments["coeff"].append(dy / dx)
                    segments["low_"].append(self._y_data[i])

                s = len(segments["low"])
                self._model.z = Var(range(s), within=Binary)
                self._model.q = Var(range(s), within=Reals)

                # -> segment selection
                self._model.choose_segment = Constraint(
                    expr=quicksum(self._model.z[k] for k in range(s)) == 1
                )

                self._model.choose_segment_low = ConstraintList()
                self._model.choose_segment_up = ConstraintList()
                for k in range(s):
                    self._model.choose_segment_low.add(
                        expr=self._model.q[k] >= segments["low"][k] * self._model.z[k]
                    )
                    self._model.choose_segment_up.add(
                        expr=self._model.q[k] <= segments["up"][k] * self._model.z[k]
                    )

                self._model.benefit_fct = Constraint(
                    expr=quicksum(
                        segments["low_"][k] * self._model.z[k]
                        + segments["coeff"][k]
                        * (self._model.q[k] - segments["low"][k] * self._model.z[k])
                        for k in range(s)
                    )
                    == self._model.benefit
                )
                self._model.capacity_ct = Constraint(
                    expr=quicksum(self._model.q[k] for k in range(s))
                    == self._model.capacity
                )

            elif self._solver_type == "gurobi":
                self._model.n_soc = Piecewise(
                    self._model.benefit,
                    self._model.capacity,
                    pw_pts=self._x_data,
                    f_rule=self._y_data,
                    pw_constr_type="EQ",
                    pw_repn="SOS2",
                )
        else:
            self._model.benefit_fct = Constraint(
                expr=self._model.benefit == self.price_limit * self._model.capacity
            )

        # -> limit maximal charging power
        self._model.power_limit = ConstraintList()

        max_power_sum = 0
        for key, car in self.cars.items():
            usage = car.get(CarData.usage, slice(d_time, None)).values
            for t in steps:
                if usage[t] > 0:
                    self._model.power_limit.add(self._model.power[key, t] <= 0)
                else:
                    self._model.power_limit.add(
                        self._model.power[key, t] <= car.maximal_charging_power
                    )

            soc = (car.soc * car.capacity) - self.dt * demand[key].sum()
            if soc < car.get_limit(d_time + td(days=1), strategy):
                max_power_sum += car.maximal_charging_power

        self._model.grid_power_limit = ConstraintList()

        for t in steps:
            self._model.grid_power_limit.add(self._model.grid[t] <= max_power_sum)

        # -> set range for soc
        self._model.soc_limit = ConstraintList()
        for key, car in self.cars.items():
            for t in self.t:
                balance = self.dt * (self._model.power[key, t] - demand[key][t])
                if t > 0:
                    self._model.soc_limit.add(
                        self._model.volume[key, t]
                        == self._model.volume[key, t - 1] + balance
                    )
                else:
                    capacity = car.soc * car.capacity
                    self._model.soc_limit.add(
                        self._model.volume[key, t] == capacity + balance
                    )

                self._model.soc_limit.add(self._model.volume[key, t] >= 0)
                self._model.soc_limit.add(self._model.volume[key, t] <= car.capacity)

        self._model.total_capacity = Constraint(
            expr=self._model.capacity
            == quicksum(self._model.volume[key, self.T - 1] for key in self.cars.keys())
        )
        # -> balance charging, pv and grid consumption
        self._model.balance = ConstraintList()
        for t in steps:
            self._model.balance.add(
                quicksum(self._model.power[key, t] for key in self.cars.keys())
                == self._model.grid[t] + self._model.pv[t]
            )

        # -> pv range
        self._model.pv_limit = ConstraintList()
        for t in steps:
            self._model.pv_limit.add(self._model.pv[t] <= generation[t])

        self._model.obj = Objective(
            expr=self._model.benefit
            - quicksum(prices[t] * self._model.grid[t] * self.dt for t in steps),
            sense=maximize,
        )

        time_range = pd.date_range(
            start=d_time, periods=len(steps), freq=RESOLUTION[self.T]
        )
        self._request = pd.Series(data=np.zeros(len(steps)), index=time_range)
        self._car_power = {
            key: pd.Series(data=np.zeros(len(steps)), index=time_range)
            for key in self.cars.keys()
        }

        used_pv = np.zeros(self.T)

        try:
            self._solver.solve(self._model)

            self._request.loc[time_range] = np.round(
                np.asarray([self._model.grid[t].value for t in steps]), 2
            )

            if "Soc" in self._used_strategy:
                self._benefit_value = value(self._model.benefit) - np.interp(
                    capacity, self._x_data, self._y_data
                )
            else:
                self._benefit_value = (
                    self.price_limit * self._request.values.sum() * self.dt
                )

            for key in self.cars.keys():
                self._car_power[key].loc[time_range] = [
                    self._model.power[key, t].value for t in steps
                ]

            if self._initial_plan:
                self._data.loc[
                    time_range, "planned_grid_consumption"
                ] = self._request.loc[time_range].copy()
                self._data.loc[time_range, "planned_pv_consumption"] = [
                    self._model.pv[t].value for t in steps
                ]
                for key, car in self.cars.items():
                    car.set_planned_charging(self._car_power[key])
                self._initial_plan = False

            used_pv = [self._model.pv[t].value for t in steps]

        except Exception as e:
            logger.debug(f" -> model infeasible {repr(e)}")
            # print(self._request.sum())

        if self._request.sum() == 0:
            self._commit = time_range[-1]
            for key, car in self.cars.items():
                car.set_final_charging(self._car_power[key])
            self._data.loc[time_range, "final_pv_consumption"] = used_pv

            self._max_requests = 5
            self._finished = True
            self._initial_plan = True

    def _plan_without_photovoltaic(self, d_time: datetime, strategy: str = "required"):
        remaining_steps = min(len(self.time_range[self.time_range >= d_time]), self.T)
        generation = self._data.loc[
            d_time : d_time + td(hours=(remaining_steps - 1) * self.dt),
            "residual_generation",
        ]
        self._request = pd.Series(
            data=np.zeros(remaining_steps),
            index=pd.date_range(
                start=d_time, freq=RESOLUTION[self.T], periods=remaining_steps
            ),
        )

        for key, car in self.cars.items():
            self._car_power[key] = pd.Series(
                data=np.zeros(remaining_steps),
                index=pd.date_range(
                    start=d_time, freq=RESOLUTION[self.T], periods=remaining_steps
                ),
            )
            usage = car.get(CarData.usage, slice(d_time, None))
            if car.soc < car.get_limit(d_time, strategy) and usage.at[d_time] == 0:
                chargeable = usage.loc[usage == 0]
                # -> get first time stamp of next charging block
                if chargeable.empty:
                    t1 = self.time_range[-1]
                else:
                    t1 = chargeable.index[0]
                # -> get first time stamp of next using block
                car_in_use = usage.loc[usage == 1]
                if car_in_use.empty:
                    t2 = self.time_range[-1]
                else:
                    t2 = car_in_use.index[0]

                if t2 > t1:
                    limit_by_capacity = (
                        (car.capacity * (1 - car.soc))
                        / car.maximal_charging_power
                        / self.dt
                    )
                    limit_by_slot = len(
                        self.time_range[
                            (self.time_range >= t1) & (self.time_range < t2)
                        ]
                    )
                    duration = int(min(limit_by_slot, limit_by_capacity))
                    self._car_power[key] = pd.Series(
                        data=car.maximal_charging_power * np.ones(duration),
                        index=pd.date_range(
                            start=d_time, freq=RESOLUTION[self.T], periods=duration
                        ),
                    )

                    self._request.loc[self._car_power[key].index] += self._car_power[
                        key
                    ].values

        if self._request.sum() > 0:
            pv_usage = pd.Series(
                data=np.zeros(remaining_steps),
                index=pd.date_range(
                    start=d_time, freq=RESOLUTION[self.T], periods=remaining_steps
                ),
            )

            generation.loc[generation.values > max(self._request.values)] = max(
                self._request.values
            )
            pv_usage.loc[self._request > 0] = generation.loc[self._request > 0].values

            self._request.loc[self._request > 0] -= generation.loc[self._request > 0]

            if self._initial_plan:
                self._initial_plan = False
                self._data.loc[
                    self._request.index, "planned_grid_consumption"
                ] = self._request.values.copy()
                self._data.loc[
                    pv_usage.index, "planned_pv_consumption"
                ] = pv_usage.copy()
                for key, car in self.cars.items():
                    car.set_planned_charging(self._car_power[key].copy())

            self._benefit_value = (
                self.price_limit * self._request.values.sum() * self.dt
            )
            self._request = self._request.loc[self._request.values > 0]

    def get_request(self, d_time: datetime, strategy: str = "MaxPvCap"):
        self._used_strategy = strategy
        if self._total_capacity > 0 and d_time > self._commit:
            if "MaxPv" in strategy:
                self._optimize_photovoltaic_usage(d_time=d_time)
            elif "PlugIn" in strategy:
                self._plan_without_photovoltaic(d_time=d_time)
            else:
                logger.error(f"invalid strategy {strategy}")
                raise Exception(f"invalid strategy {strategy}")

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
            if "MaxPv" in self._used_strategy:
                self._finished = True
            self._initial_plan = True
            return True
        else:
            if "MaxPv" in self._used_strategy:
                self._data.loc[price.index, "grid_fee"] = price.values
            else:
                self._commit += td(minutes=np.random.randint(low=1, high=3))
            self._max_requests -= 1

            return False
