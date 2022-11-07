from datetime import datetime
from datetime import timedelta as td
import pandas as pd
import numpy as np
import logging

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
    Param
)

from carLib.car import Car, CarData

logger = logging.getLogger("samrtdso.energy_system_dispatch")


class EnergySystemDispatch:
    def __init__(self, steps: int = 96,
                 resolution: str = '15min',
                 strategy: str = 'soc',
                 benefit_function: pd.Series = None,
                 electric_vehicles: list[Car] = None,
                 generation: pd.Series = None,
                 tariff: pd.Series = None,
                 grid_fee: pd.Series = None,
                 solver: str = 'glpk'):

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

        maximal_capacity = sum([car.capacity for car in self.ev])
        self.soc_s = list(benefit_function.index / 100 * maximal_capacity)
        self.benefits = list(np.cumsum(benefit_function.values * maximal_capacity * 0.05))

        self.strategy = strategy
        if self.strategy == 'soc':
            self.segments = dict(low=[], up=[], coeff=[], low_=[])
            for i in range(0, len(self.benefits) - 1):
                self.segments["low"].append(self.soc_s[i])
                self.segments["up"].append(self.soc_s[i + 1])
                dy = self.benefits[i + 1] - self.benefits[i]
                dx = self.soc_s[i + 1] - self.soc_s[i]
                self.segments["coeff"].append(dy / dx)
                self.segments["low_"].append(self.benefits[i])
        else:
            self.price_limit = benefit_function.values[0]

        self.request = None
        self.pv_charge = None
        self.benefit = 0

    def get_actual_ev_capacity(self, ev: Car = None):
        capacity = 0
        if ev is None:
            for car in self.ev:
                capacity += car.get_current_capacity()
            return capacity
        else:
            return ev.get_current_capacity()

    def get_actual_ev_benefit(self):
        c = self.get_actual_ev_capacity()
        b = np.interp(c, self.soc_s, self.benefits)
        return b

    def get_ev_soc_limit(self, d_time: datetime, ev: Car):
        s1, s2 = d_time, d_time + td(days=1)
        return ev.get_limit(s2, strategy='required')

    def get_maximal_ev_power(self, ev: Car = None):
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
        time_range = pd.date_range(start=s1, periods=self.T, freq=self.resolution)

        generation = self.generation.loc[slice(s1, s2)].values

        tariff = self.tariff.loc[slice(s1, s2)].values.flatten()
        grid_fee = self.grid_fee.loc[slice(s1, s2)].values.flatten()
        total_price = tariff + grid_fee
        # -> clear model
        self.m.clear()

        # -> declare variables
        self.m.power = Var(self.num_ev, self.t, within=Reals, bounds=(0, None))
        self.m.grid = Var(self.t, within=Reals, bounds=(0, None))
        self.m.pv = Var(self.t, within=Reals, bounds=(0, None))
        self.m.capacity = Var(within=Reals, bounds=(0, self.get_maximal_ev_capacity()))
        self.m.volume = Var(self.num_ev, self.t)
        self.m.benefit = Var(initialize=self.get_actual_ev_benefit())

        self.m.capacity_eq = ConstraintList()
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

            self.m.benefit_eq = Constraint(expr=self.m.benefit == quicksum(self.segments["low_"][k] * self.m.z[k]
                                                                           + self.segments["coeff"][k]
                                                                           * (self.m.q[k] - self.segments["low"][k] *
                                                                              self.m.z[k])
                                                                           for k in s))
            self.m.capacity_eq.add(self.m.capacity == quicksum(self.m.q[k] for k in s))

        else:
            self.m.benefit_eq = Constraint(expr=self.m.benefit == self.price_limit * self.m.capacity)

        # -> limit maximal charging power
        self.m.power_limit = ConstraintList()
        # -> limit volume to range
        self.m.capacity_limit = ConstraintList()
        # -> max value for grid supply
        total_grid_power = 0
        for i, car in zip(self.num_ev, self.ev):
            usage = self.get_ev_usage(d_time, car)
            demand = self.get_ev_demand(d_time, car)

            end_of_day_capacity = self.get_actual_ev_capacity(car) - demand.sum()
            end_of_day_soc = end_of_day_capacity / self.get_maximal_ev_capacity(car)

            power = self.get_maximal_ev_power(car)
            volume = self.get_actual_ev_capacity(car)

            if end_of_day_soc < self.get_ev_soc_limit(d_time, car):
                total_grid_power += self.get_maximal_ev_power(car)
            # -> set power and volume limits
            for t in self.t:
                # -> set power limit
                self.m.power_limit.add(self.m.power[i, t] <= (1 - usage[t]) * power)
                # -> set volume limit
                self.m.capacity_limit.add(self.m.volume[i, t] >= 0)
                self.m.capacity_limit.add(self.m.volume[i, t] <= self.get_maximal_ev_capacity(car))
                in_out = self.m.power[i, t] * self.dt - demand[t]
                if t > 0:
                    volume = self.m.volume[i, t - 1]
                self.m.capacity_limit.add(self.m.volume[i, t] == in_out + volume)

        self.m.grid_power_limit = ConstraintList()
        for t in self.t:
            self.m.grid_power_limit.add(self.m.grid[t] <= total_grid_power)

        self.m.capacity_eq.add(self.m.capacity == quicksum(self.m.volume[:, self.T - 1]))

        # -> set grid consumption
        self.m.grid_limit = ConstraintList()
        for t in self.t:
            self.m.grid_limit.add(self.m.grid[t] == - self.m.pv[t] + quicksum(self.m.power[:, t]))

        # -> limit pv range
        self.m.pv_limit = ConstraintList()
        for t in self.t:
            self.m.pv_limit.add(self.m.pv[t] <= generation[t])

        costs = quicksum(total_price[t] * self.m.grid[t] * self.dt for t in self.t)

        self.m.obj = Objective(expr=self.m.benefit - costs, sense=maximize)

        try:
            self.s.solve(self.m)
            grid_consumption = np.array([self.m.grid[t].value for t in self.t])
            pv_charge = np.array([self.m.pv[t].value for t in self.t])

            for i, car in zip(self.num_ev, self.ev):
                car_charging = np.array([self.m.power[i, t].value for t in self.t])
                car_charging = pd.Series(data=car_charging, index=time_range)
                car.set_planned_charging(car_charging)

            if self.strategy == 'soc':
                benefit = value(self.m.benefit) - self.get_actual_ev_benefit()
            else:
                benefit = value(self.m.benefit) - self.price_limit * self.get_actual_ev_capacity()

            self.benefit = benefit
            self.request = pd.Series(data=grid_consumption, index=time_range)
            self.pv_charge = pd.Series(data=pv_charge, index=time_range)

        except Exception as e:
            logger.warning(f"can not solve optimization problem")
            logger.warning(f"{repr(e)}")

# def _plan_without_photovoltaic(self, d_time: datetime, strategy: str = "required"):
#     remaining_steps = min(len(self.time_range[self.time_range >= d_time]), self.T)
#     generation = self._data.loc[
#                  d_time: d_time + td(hours=(remaining_steps - 1) * self.dt),
#                  "residual_generation",
#                  ]
#     self._request = pd.Series(
#         data=np.zeros(remaining_steps),
#         index=pd.date_range(
#             start=d_time, freq=RESOLUTION[self.T], periods=remaining_steps
#         ),
#     )
#
#     for key, car in self.cars.items():
#         self._car_power[key] = pd.Series(
#             data=np.zeros(remaining_steps),
#             index=pd.date_range(
#                 start=d_time, freq=RESOLUTION[self.T], periods=remaining_steps
#             ),
#         )
#         usage = car.get(CarData.usage, slice(d_time, None))
#         if car.soc < car.get_limit(d_time, strategy) and usage.at[d_time] == 0:
#             chargeable = usage.loc[usage == 0]
#             # -> get first time stamp of next charging block
#             if chargeable.empty:
#                 t1 = self.time_range[-1]
#             else:
#                 t1 = chargeable.index[0]
#             # -> get first time stamp of next using block
#             car_in_use = usage.loc[usage == 1]
#             if car_in_use.empty:
#                 t2 = self.time_range[-1]
#             else:
#                 t2 = car_in_use.index[0]
#
#             if t2 > t1:
#                 limit_by_capacity = (
#                         (car.capacity * (1 - car.soc))
#                         / car.maximal_charging_power
#                         / self.dt
#                 )
#                 limit_by_slot = len(
#                     self.time_range[
#                         (self.time_range >= t1) & (self.time_range < t2)
#                         ]
#                 )
#                 duration = int(min(limit_by_slot, limit_by_capacity))
#                 self._car_power[key] = pd.Series(
#                     data=car.maximal_charging_power * np.ones(duration),
#                     index=pd.date_range(
#                         start=d_time, freq=RESOLUTION[self.T], periods=duration
#                     ),
#                 )
#
#                 self._request.loc[self._car_power[key].index] += self._car_power[
#                     key
#                 ].values
#
#     if self._request.sum() > 0:
#         pv_usage = pd.Series(
#             data=np.zeros(remaining_steps),
#             index=pd.date_range(
#                 start=d_time, freq=RESOLUTION[self.T], periods=remaining_steps
#             ),
#         )
#
#         generation.loc[generation.values > max(self._request.values)] = max(
#             self._request.values
#         )
#         pv_usage.loc[self._request > 0] = generation.loc[self._request > 0].values
#
#         self._request.loc[self._request > 0] -= generation.loc[self._request > 0]
#
#         if self._initial_plan:
#             self._initial_plan = False
#             self._data.loc[
#                 self._request.index, "planned_grid_consumption"
#             ] = self._request.values.copy()
#             self._data.loc[
#                 pv_usage.index, "planned_pv_consumption"
#             ] = pv_usage.copy()
#             for key, car in self.cars.items():
#                 car.set_planned_charging(self._car_power[key].copy())
#
#         self._benefit_value = (
#                 self.price_limit * self._request.values.sum() * self.dt
#         )
#         self._request = self._request.loc[self._request.values > 0]
