from datetime import datetime
from datetime import timedelta as td
import pandas as pd
import numpy as np
import logging

from pyomo.environ import (
    ConcreteModel,
    Constraint,
    ConstraintList,
    Objective,
    Reals,
    Binary,
    SolverFactory,
    Var,
    maximize,
    quicksum,
    value,
)

from pyomo.opt import (
    SolverStatus,
    TerminationCondition
)

from carLib.car import Car, CarData
from participants.utils import HeatStorage

logger = logging.getLogger("samrtdso.energy_system_dispatch")
logger.setLevel('ERROR')


class EnergySystemDispatch:
    def __init__(self, steps: int = 96,
                 resolution: str = '15min',
                 strategy: str = 'soc',
                 benefit_functions: pd.DataFrame = None,
                 electric_vehicles: dict[str, Car] = None,
                 heat_storages: dict[str, HeatStorage] = None,
                 heat_pump: float = 25,
                 generation: pd.Series = None,
                 heat_demand: pd.DataFrame = None,
                 tariff: pd.Series = None,
                 grid_fee: pd.Series = None,
                 solver: str = 'gurobi',
                 analyse_ev: bool = True,
                 analyse_hp: bool = True):

        self.resolution = resolution
        self.T = steps
        self.t = range(steps)
        self.dt = 1 / (steps / 24)

        self.heat_storages = heat_storages
        self.heat_demand = heat_demand
        self.heat_pump = heat_pump

        self.ev = electric_vehicles

        self.generation = generation

        self.tariff = tariff
        self.grid_fee = grid_fee

        self.m = ConcreteModel()
        self.s = SolverFactory(solver)
        self._analyse_ev = analyse_ev
        self._analyse_hp = analyse_hp

        self.soc_s = {}
        self.benefits = {}
        self.segments = {}

        for key, car in self.ev.items():
            self.strategy = strategy
            self.soc_s[key] = list(benefit_functions.index / 100 * car.capacity)
            self.benefits[key] = list(np.cumsum(benefit_functions[key].values * car.capacity * 0.05))
            if "soc" in self.strategy:
                self.segments[key] = dict(low=[], up=[], coeff=[], low_=[])
                for i in range(0, len(self.benefits[key]) - 1):
                    self.segments[key]["low"].append(self.soc_s[key][i])
                    self.segments[key]["up"].append(self.soc_s[key][i + 1])
                    dy = self.benefits[key][i + 1] - self.benefits[key][i]
                    dx = self.soc_s[key][i + 1] - self.soc_s[key][i]
                    self.segments[key]["coeff"].append(dy / dx)
                    self.segments[key]["low_"].append(self.benefits[key][i])
            else:
                self.price_limit = benefit_functions.values[0]

        self.request = None
        self.pv_usage = None
        self.ev_benefit = 0
        self.hp_benefit = 0
        self.power_demand_heat_s = np.zeros(self.T)
        self.power_demand_heat_c = np.zeros(self.T)
        self.power_demand_mobility = np.zeros(self.T)
        self.power_generation_pv = np.zeros(self.T)
        self.heat_cost_with_storage = 0
        self.pv_benefit = 0
        self.grid_out = pd.Series(dtype=float)
        self.grid_in = pd.Series(dtype=float)

    def get_actual_ev_capacity(self, ev: Car = None):
        capacity = 0
        if ev is None:
            for car in self.ev.values():
                capacity += car.get_current_capacity()
            return capacity
        else:
            return ev.get_current_capacity()

    def get_actual_ev_benefit(self, ev: str):
        c = self.get_actual_ev_capacity(self.ev[ev])
        b = np.interp(c, self.soc_s[ev], self.benefits[ev])
        return b

    def get_ev_soc_limit(self, d_time: datetime, ev: Car):
        s1, s2 = d_time, d_time + td(days=1)
        return ev.get_limit(s2, strategy='required')

    def get_maximal_ev_power(self, ev: Car = None):
        power = 0
        if ev is None:
            for car in self.ev.values():
                power += car.maximal_charging_power
            return power
        else:
            return ev.maximal_charging_power

    def get_maximal_ev_capacity(self, ev: Car = None):
        capacity = 0
        if ev is None:
            for car in self.ev.values():
                capacity += car.capacity
            return capacity
        else:
            return ev.capacity

    def get_ev_demand(self, d_time: datetime, ev: Car = None):
        s1, s2 = d_time, d_time + td(days=1)
        if ev is None:
            demand = np.zeros(self.T)
            for car in self.ev.values():
                demand += car.get(CarData.demand, slice(s1, s2)).values
            return demand
        else:
            return ev.get(CarData.demand, slice(s1, s2)).values

    def get_ev_usage(self, d_time: datetime, ev: Car):
        s1, s2 = d_time, d_time + td(days=1)
        return ev.get(CarData.usage, slice(s1, s2)).values

    def _get_time_slice(self, d_time: datetime):
        s1, s2 = d_time, d_time + td(hours=23, minutes=59, seconds=59)
        time_range = pd.date_range(start=s1, periods=self.T, freq=self.resolution)
        return slice(s1, s2), time_range

    def _create_ev_model(self, d_time: datetime):

        # -> declare variables
        self.m.ev_power = Var(self.ev.keys(), self.t, within=Reals, bounds=(0, None))
        self.m.ev_benefit = Var(self.ev.keys(), self.t, within=Reals, bounds=(None, None))

        self.m.ev_volume = Var(self.ev.keys(), self.t)
        self.m.ev_capacity_eq = ConstraintList()
        self.m.benefit_eq = ConstraintList()

        # -> build soc dependent benefit function
        if 'soc' in self.strategy:
            s = range(len(self.segments[list(self.ev.keys())[0]]["low"]))

            self.m.z = Var(self.ev.keys(), self.t, s, within=Binary)
            self.m.q = Var(self.ev.keys(), self.t, s, within=Reals)

            self.m.choose_segment_eq = ConstraintList()
            self.m.s_segment_low = ConstraintList()
            self.m.s_segment_up = ConstraintList()

            for car in self.ev.keys():

                # -> segment selection
                for t in self.t:
                    self.m.choose_segment_eq.add(quicksum(self.m.z[car, t, k] for k in s) == 1)
                    for k in s:
                        self.m.s_segment_low.add(self.m.q[car, t, k] >=
                                                 self.segments[car]["low"][k] * self.m.z[car, t, k])
                        self.m.s_segment_up.add(self.m.q[car, t, k] <=
                                                self.segments[car]["up"][k] * self.m.z[car, t, k])

                    self.m.benefit_eq.add(self.m.ev_benefit[car, t] ==
                                          quicksum(self.segments[car]["low_"][k] * self.m.z[car, t, k]
                                                   + self.segments[car]["coeff"][k]
                                                   * (self.m.q[car, t, k] - self.segments[car]["low"][k] *
                                                      self.m.z[car, t, k])
                                                   for k in s))
                    self.m.ev_capacity_eq.add(self.m.ev_volume[car, t] == quicksum(self.m.q[car, t, k] for k in s))

        # -> build simple benefit function
        else:
            for car in self.ev.keys():
                for t in self.t:
                    self.m.benefit_eq.add(self.m.ev_benefit[car, t] == self.price_limit * self.m.ev_volume[car, t])

        self.m.ev_dbenefit_eq = ConstraintList()

        self.m.ev_p_dbenefit = Var(self.ev.keys(), self.t, within=Reals, bounds=(0, None))
        self.m.ev_m_dbenefit = Var(self.ev.keys(), self.t, within=Reals, bounds=(0, None))

        self.m.ev_dbenefit = Var(self.ev.keys(), self.t, within=Reals, bounds=(None, None))

        for car, val in self.ev.items():
            for t in self.t:
                if t == 0:
                    d_benefit = self.m.ev_benefit[car, t] - self.get_actual_ev_benefit(car)
                else:
                    d_benefit = self.m.ev_benefit[car, t] - self.m.ev_benefit[car, t - 1]
                self.m.ev_dbenefit_eq.add(self.m.ev_dbenefit[car, t] == d_benefit)

                self.m.ev_dbenefit_eq.add(self.m.ev_dbenefit[car, t] == self.m.ev_p_dbenefit[car, t] -
                                          self.m.ev_m_dbenefit[car, t])

        # -> limit maximal charging power
        self.m.ev_power_limit = ConstraintList()
        # -> limit volume to range
        self.m.ev_capacity_limit = ConstraintList()
        # -> max value for grid supply for evs
        total_grid_power = 0
        for i, car in self.ev.items():
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
                self.m.ev_power_limit.add(self.m.ev_power[i, t] <= (1 - usage[t]) * power)
                # -> set volume limit
                self.m.ev_capacity_limit.add(self.m.ev_volume[i, t] >= 0)
                self.m.ev_capacity_limit.add(self.m.ev_volume[i, t] <= self.get_maximal_ev_capacity(car))
                in_out = self.m.ev_power[i, t] * self.dt - demand[t]
                if t > 0:
                    volume = self.m.ev_volume[i, t - 1]
                self.m.ev_capacity_limit.add(self.m.ev_volume[i, t] == in_out + volume)

        return total_grid_power

    def _create_hp_model(self, d_time: datetime):
        sl, time_range = self._get_time_slice(d_time)

        # -> declare variables
        self.m.hp_heat = Var(self.heat_storages.keys(), self.t, within=Reals, bounds=(0, None))
        self.m.hs_volume = Var(self.heat_storages.keys(), self.t, within=Reals, bounds=(0, None))
        self.m.hs_in = Var(self.heat_storages.keys(), self.t, within=Reals, bounds=(0, None))
        self.m.hs_out = Var(self.heat_storages.keys(), self.t, within=Reals, bounds=(0, None))

        heat_demand, heat_cop = {}, {}

        for key in self.heat_demand.columns:
            if 'heat' in key:
                k = key.replace('heat_', '')
                heat_demand[k] = self.heat_demand[key].loc[sl].values
            elif 'COP' in key:
                k = key.replace('COP_', '')
                heat_cop[k] = self.heat_demand[key].loc[sl].values

        self.m.heat_balance_eq = ConstraintList()
        self.m.heat_capacity_eq = ConstraintList()
        for t in self.t:
            for key in heat_demand.keys():
                loss = (self.heat_storages[key].loss / self.T)
                self.m.heat_balance_eq.add(heat_demand[key][t] + loss + self.m.hs_in[key, t] == self.m.hs_out[key, t]
                                           + self.m.hp_heat[key, t])
                if t > 0:
                    self.m.heat_capacity_eq.add(self.m.hs_volume[key, t] == self.m.hs_volume[key, t - 1]
                                                + (self.m.hs_in[key, t] - self.m.hs_out[key, t]) * self.dt)
                else:
                    self.m.heat_capacity_eq.add(self.m.hs_volume[key, t] == self.heat_storages[key].V0
                                                + (self.m.hs_in[key, t] - self.m.hs_out[key, t]) * self.dt)

                self.m.heat_capacity_eq.add(self.m.hs_volume[key, t] >= 0)
                self.m.heat_capacity_eq.add(self.m.hs_volume[key, t] <= self.heat_storages[key].volume)

            self.m.heat_balance_eq.add(quicksum(self.m.hp_heat[key, t] for key in heat_demand.keys())
                                       <= self.heat_pump)

        for key in heat_demand.keys():
            self.m.heat_capacity_eq.add(self.m.hs_volume[key, self.T - 1] == self.heat_storages[key].volume / 2)

        return heat_demand, heat_cop

    def get_optimal_solution(self, d_time: datetime):
        # -> clear model
        self.m.clear()
        sl, time_range = self._get_time_slice(d_time)
        # -> get economic time series
        tariff = self.tariff.loc[sl].values.flatten()
        grid_fee = self.grid_fee.loc[sl].values.flatten()
        total_price = tariff + grid_fee
        # -> get generation time series
        generation = self.generation.loc[sl].values
        # -> build model for ev
        if self._analyse_ev:
            total_grid_power = self._create_ev_model(d_time)
            ev_power = self.m.find_component('ev_power')
            ev_benefit = quicksum(self.m.ev_dbenefit[car, t] for car in self.ev.keys() for t in self.t)
        else:
            total_grid_power = 0
            ev_power = np.zeros((1, self.T))
            ev_benefit = 0

        # -> build model for hp
        if self._analyse_hp:
            heat_demand, heat_cop = self._create_hp_model(d_time)

            hp_power = [quicksum(self.m.hp_heat[key, t] / heat_cop[key][t] for key in heat_cop.keys())
                        for t in self.t]

            # power = quicksum(self.m.hs_volume[key, self.T - 1] / np.mean(heat_cop[key])
            #                  for key in heat_cop.keys())
            # heat_demand_s = [quicksum((heat_demand[key][t] / heat_cop[key][t]) for key in heat_cop.keys())
            #                  for t in self.t]
        else:
            hp_power = np.zeros(self.T)
            heat_demand_s = np.zeros(self.T)

        # -> declare grid variables
        self.m.grid_power = Var(self.t, within=Reals, bounds=(0, None))
        self.m.grid_in = Var(self.t, within=Reals, bounds=(0, None))
        self.m.grid_out = Var(self.t, within=Reals, bounds=(0, None))

        self.m.power_balance_eq = ConstraintList()
        for t in self.t:
            # -> sum of grid input and output
            self.m.power_balance_eq.add(self.m.grid_power[t] == self.m.grid_out[t] - self.m.grid_in[t])
            # -> balance equation at input node
            self.m.power_balance_eq.add(self.m.grid_out[t] == (hp_power[t] + quicksum(ev_power[:, t])
                                                               + self.m.grid_in[t]) - generation[t])

            # -> limit grid consumption to ev charging and heat pump consumption
            self.m.power_balance_eq.add(self.m.grid_out[t] <= total_grid_power + hp_power[t])

        costs = quicksum(total_price[t] * self.m.grid_out[t] * self.dt for t in self.t)
        revenue = quicksum(5 * self.m.grid_in[t] * self.dt for t in self.t)

        self.m.obj = Objective(expr=ev_benefit - costs + revenue, sense=maximize)

        try:

            results = self.s.solve(self.m)
            status = results.solver.status

            termination = results.solver.termination_condition
            if (status == SolverStatus.ok) and (termination == TerminationCondition.optimal):
                logger.info(f" -> found optimal solution")

                grid_consumption = np.array([self.m.grid_out[t].value for t in self.t])
                grid_feed_in = np.array([self.m.grid_in[t].value for t in self.t])
                pv_usage = np.array([generation[t] - self.m.grid_in[t].value for t in self.t])
                if self._analyse_ev:
                    for i, car in self.ev.items():
                        car_charging = np.array([self.m.ev_power[i, t].value for t in self.t])
                        car_charging = pd.Series(data=car_charging, index=time_range)
                        car.set_planned_charging(car_charging)

                    benefit = [value(self.m.ev_dbenefit[car, t]) for car in self.ev.keys() for t in self.t]
                    benefit = np.array(benefit)
                    benefit[benefit < 0] = 0
                    # -> set benefit value to compare it later with total charging costs
                    self.ev_benefit = np.round(benefit.sum(), 2)

                if self._analyse_hp:
                    for i, storage in self.heat_storages.items():
                        storage_volume = np.array([self.m.hs_volume[i, t].value for t in self.t])
                        storage.set_planned_usage(storage_volume)

                    # -> set benefit value to compare it later with total heating costs
                    # self.hp_benefit = value(hs_benefit)
                    self.power_demand_heat_s = np.array([value(hp_power[t]) for t in self.t])
                    # self.power_demand_heat_c = np.array([value(heat_demand_s[t]) for t in self.t])

                self.pv_benefit = value(revenue)

                # max_hp = np.vstack([self.power_demand_heat_s, self.power_demand_heat_c]).max(axis=0, initial=0)
                # self.power_demand_mobility = np.array([value(quicksum(ev_power[:, t])) for t in self.t])
                # self.power_generation_pv = generation

                request_power = self.power_demand_heat_s + self.power_demand_mobility - self.power_generation_pv

                # -> set request series which is send to the Capacity Provider
                self.request = pd.Series(data=request_power, index=time_range)
                self.request = self.request.round(2)
                # -> set pv charging time series
                self.pv_usage = pd.Series(data=pv_usage, index=time_range)
                self.pv_usage = self.pv_usage.round(2)
                # -> set grid consumption and grid feed in
                self.grid_out = pd.Series(data=grid_consumption, index=time_range)
                self.grid_in = pd.Series(data=grid_feed_in, index=time_range)

            elif termination == TerminationCondition.infeasible:
                raise Exception(' -> model infeasible')
            else:
                raise Exception(' -> model error')

        except Exception as e:
            logger.warning(f"can not solve optimization problem")
            logger.warning(f"{repr(e)}")
            # -> set benefit to zero
            self.hp_benefit = 0
            self.ev_benefit = 0
            self.pv_benefit = 0
            # -> set grid consumption and pv charging to zero
            self.request = pd.Series(data=np.zeros(self.T), index=time_range)
            self.pv_usage = pd.Series(data=np.zeros(self.T), index=time_range)
            self.grid_out = pd.Series(data=np.zeros(self.T), index=time_range)
            self.grid_in = pd.Series(data=np.zeros(self.T), index=time_range)
            self.power_demand_heat_c = np.zeros(self.T)
            self.power_demand_heat_s = np.zeros(self.T)
            self.power_demand_mobility = np.zeros(self.T)
            self.power_generation_pv = np.zeros(self.T)

    def get_heuristic_solution(self, d_time: datetime):
        pass
#         remaining_steps = min(len(self.time_range[self.time_range >= d_time]), self.T)
#         generation = self._data.loc[
#                      d_time: d_time + td(hours=(remaining_steps - 1) * self.dt),
#                      "residual_generation",
#                      ]
#         self._request = pd.Series(
#             data=np.zeros(remaining_steps),
#             index=pd.date_range(
#                 start=d_time, freq=RESOLUTION[self.T], periods=remaining_steps
#             ),
#         )
#
#         for key, car in self.cars.items():
#             self._car_power[key] = pd.Series(
#                 data=np.zeros(remaining_steps),
#                 index=pd.date_range(
#                     start=d_time, freq=RESOLUTION[self.T], periods=remaining_steps
#                 ),
#             )
#             usage = car.get(CarData.usage, slice(d_time, None))
#             if car.soc < car.get_limit(d_time, strategy) and usage.at[d_time] == 0:
#                 chargeable = usage.loc[usage == 0]
#                 # -> get first time stamp of next charging block
#                 if chargeable.empty:
#                     t1 = self.time_range[-1]
#                 else:
#                     t1 = chargeable.index[0]
#                 # -> get first time stamp of next using block
#                 car_in_use = usage.loc[usage == 1]
#                 if car_in_use.empty:
#                     t2 = self.time_range[-1]
#                 else:
#                     t2 = car_in_use.index[0]
#
#                 if t2 > t1:
#                     limit_by_capacity = (
#                             (car.capacity * (1 - car.soc))
#                             / car.maximal_charging_power
#                             / self.dt
#                     )
#                     limit_by_slot = len(
#                         self.time_range[
#                             (self.time_range >= t1) & (self.time_range < t2)
#                             ]
#                     )
#                     duration = int(min(limit_by_slot, limit_by_capacity))
#                     self._car_power[key] = pd.Series(
#                         data=car.maximal_charging_power * np.ones(duration),
#                         index=pd.date_range(
#                             start=d_time, freq=RESOLUTION[self.T], periods=duration
#                         ),
#                     )
#
#                     self._request.loc[self._car_power[key].index] += self._car_power[
#                         key
#                     ].values
#
#         if self._request.sum() > 0:
#             pv_usage = pd.Series(
#                 data=np.zeros(remaining_steps),
#                 index=pd.date_range(
#                     start=d_time, freq=RESOLUTION[self.T], periods=remaining_steps
#                 ),
#             )
#
#             generation.loc[generation.values > max(self._request.values)] = max(
#                 self._request.values
#             )
#             pv_usage.loc[self._request > 0] = generation.loc[self._request > 0].values
#
#             self._request.loc[self._request > 0] -= generation.loc[self._request > 0]
#
#             if self._initial_plan:
#                 self._initial_plan = False
#                 self._data.loc[
#                     self._request.index, "planned_grid_consumption"
#                 ] = self._request.values.copy()
#                 self._data.loc[
#                     pv_usage.index, "planned_pv_consumption"
#                 ] = pv_usage.copy()
#                 for key, car in self.cars.items():
#                     car.set_planned_charging(self._car_power[key].copy())
#
#             self._benefit_value = (
#                     self.price_limit * self._request.values.sum() * self.dt
#             )
#             self._request = self._request.loc[self._request.values > 0]
#
# # def _plan_without_photovoltaic(self, d_time: datetime, strategy: str = "required"):
#
