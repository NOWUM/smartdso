from datetime import datetime
from datetime import timedelta as td
import pandas as pd
import numpy as np

from pyomo.environ import (
    Expression,
    ConcreteModel,
    Constraint,
    ConstraintList,
    Objective,
    Piecewise,
    Reals,
    Binary,
    SolverFactory,
    Var,
    maximize,
    minimize,
    quicksum,
    value,
)

from carLib.car import Car, CarData


class EnergySystemDispatch:

    def __int__(
            self,
            steps: int = 96,
            resolution: str = '15min',
            strategy: str = 'soc',
            benefit_function: pd.Series = None,
            electric_vehicles: list[Car] = None,
            generation: pd.Series = None,
            tariff: pd.Series = None,
            grid_fee: pd.Series = None,
            solver: str = 'glpk'
    ):

        self.resolution = resolution
        self.T = steps
        self.t = range(steps)
        self.dt = 1 / (steps / 24)

        self.ev = electric_vehicles
        self.num_ev = range(len(self.ev))

        self.generation = generation

        self.tariff = tariff
        self.grid_fee = grid_fee

        self.m = ConcreteModel()
        self.s = SolverFactory(solver)

        self.strategy = strategy
        if self.strategy == 'soc':
            maximal_capacity = sum([car.capacity for car in self.ev])
            soc_s = list(benefit_function.index / 100 * maximal_capacity)
            benefits = list(np.cumsum(benefit_function.values * maximal_capacity * 0.05))

            self.segments = dict(low=[], up=[], coeff=[], low_=[])
            for i in range(0, len(benefits) - 1):
                self.segments["low"].append(soc_s[i])
                self.segments["up"].append(soc_s[i + 1])
                dy = benefits[i + 1] - benefits[i]
                dx = soc_s[i + 1] - soc_s[i]
                self.segments["coeff"].append(dy / dx)
                self.segments["low_"].append(benefits[i])
        else:
            self.price_limit = benefit_function.values[0]

    def get_actual_ev_capacity(self, ev: Car = None):
        capacity = 0
        if ev is None:
            for car in self.ev:
                capacity += car.get_current_capacity()
            return capacity
        else:
            return ev.get_current_capacity()

    def get_ev_soc_limit(self, d_time: datetime, ev: Car):
        s1, s2 = d_time, d_time + td(days=1)
        return ev.get_limit(s2, strategy='required')

    def get_maximal_ev_power(self, ev:Car = None):
        power = 0
        if ev is None:
            for car in self.ev:
                power += car.maximal_charging_power
            return power
        else:
            return ev.maximal_charging_power

    def get_maximal_ev_capacity(self, ev: Car = None):
        capacity = 0
        if ev is None:
            for car in self.ev:
                capacity += car.capacity
            return capacity
        else:
            return ev.capacity

    def get_ev_demand(self, d_time: datetime, ev: Car = None):
        s1, s2 = d_time, d_time + td(days=1)
        if ev is None:
            demand = np.zeros(self.T)
            for car in self.ev:
                demand += car.get(CarData.demand, slice(s1, s2)).values
            return demand
        else:
            return ev.get(CarData.demand, slice(s1, s2)).values

    def get_ev_usage(self, d_time: datetime, ev: Car):
        s1, s2 = d_time, d_time + td(days=1)
        return ev.get(CarData.usage, slice(s1, s2)).values

    def optimal_solution(self, d_time: datetime):
        s1, s2 = d_time, d_time + td(days=1)

        ev_capacity = self.get_actual_ev_capacity()
        ev_demand = self.get_ev_demand(d_time)

        generation = self.generation.loc[slice(s1, s2)].values

        tariff = self.tariff.loc[slice(s1, s2)].values
        grid_fee = self.grid_fee.loc[slice(s1, s2)].values
        total_price = tariff + grid_fee

        # -> clear model
        self.m.clear()
        # -> declare variables
        self.m.power = Var(self.num_ev, self.t, within=Reals, bounds=(0, None))
        self.m.grid = Var(self.t, within=Reals, bounds=(0, None))
        self.m.pv = Var(self.t, within=Reals, bounds=(0, None))
        self.m.capacity = Var(within=Reals, bounds=(0, self.get_maximal_ev_capacity()))
        self.m.volume = Var(self.num_ev, self.t, within=Reals)
        self.m.benefit = Var(within=Reals, bounds=(0, None))

        if self.strategy == 'soc':
            s = range(len(self.segments["low"]))
            self.m.z = Var(s, within=Binary)
            self.m.q = Var(s, within=Reals)

            # -> segment selection
            self.m.choose_segment = Constraint(quicksum(self.m.z[k] for k in s) == 1)

            self.m.s_segment_low = ConstraintList()
            self.m.s_segment_up = ConstraintList()
            for k in s:
                self.m.s_segment_low.add(self.m.q[k] >= self.segments["low"][k] * self.m.z[k])
                self.m.s_segment_up.add(self.m.q[k] <= self.segments["up"][k] * self.m.z[k])

            self.m.benefit_fct = Constraint(quicksum(self.segments["low_"][k] * self.m.z[k]
                                                     + self.segments["coeff"][k]
                                                     * (self.m.q[k] - self.segments["low"][k] * self.m.z[k])
                                                     for k in s) == self.m.benefit)

            self.m.capacity_ct = Constraint(quicksum(self.m.q[k] for k in s) == self.m.capacity)

        else:
            self.m.benefit_fct = Constraint(self.m.benefit == self.price_limit * self.m.capacity)

        # -> limit maximal charging power
        self.m.power_limit = ConstraintList()
        # -> limit volume to range
        self.m.capacity_limit = ConstraintList()
        # -> max value for grid supply
        total_grid_power = 0
        for i, car in zip(self.num_ev, self.ev):
            usage = self.get_ev_usage(d_time, car)
            demand = self.get_ev_demand(d_time, car)
            # -> set power limits
            for t in self.t:
                if usage[t] > 0:
                    self.m.power_limit.add(self.m.power[i, t] <= 0)
                else:
                    self.m.power_limit.add(self.m.power[i, t] <= self.get_maximal_ev_power(car))

            end_of_day_capacity = self.get_actual_ev_capacity(car) - demand.sum()
            end_of_day_soc = end_of_day_capacity / self.get_maximal_ev_capacity(car)

            if end_of_day_soc < self.get_ev_soc_limit(d_time, car):
                total_grid_power += self.get_maximal_ev_power(car)
            # -> set capacity limits

            for t in self.t:
                eq = self.m.power[i, t] * self.dt - demand[t]

                if t == 0:
                    eq += self.get_actual_ev_capacity(car)
                else:
                    eq += self.m.volume[i, t - 1]

                self.m.capacity_limit.add(self.m.volume[i, t] == eq)
                self.m.soc_limit.add(self.m.volume[i, t] >= 0)
                self.m.soc_limit.add(self.m.volume[i, t] <= self.get_maximal_ev_capacity(car))

        self.m.grid_power_limit = ConstraintList()
        for t in self.t:
            self.m.grid_power_limit.add(self.m.grid[t] <= total_grid_power)



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

        if "Soc" in self.strategy:
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
                 d_time: d_time + td(hours=(remaining_steps - 1) * self.dt),
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
