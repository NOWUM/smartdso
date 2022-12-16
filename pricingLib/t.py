import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from datetime import timedelta as td

from pyomo.environ import Constraint, Var, Objective, SolverFactory, ConcreteModel, \
     Reals, Binary, maximize, value, quicksum, ConstraintList

from pyomo.opt import (
    SolverStatus,
    TerminationCondition
)

from participants.resident import Resident
from carLib.car import CarData

price_data = pd.read_pickle(r'./pricingLib/price_data.pkl')
price_data = price_data.loc[price_data.index.year == 2023]

benefit_functions = r"./participants/strategy/benefit_function.csv"
# -> benefit functions from survey
BENEFIT_FUNCTIONS = pd.read_csv(benefit_functions, index_col=0)
# -> adjust to actually market level
BENEFIT_FUNCTIONS += 0.10
BENEFIT_FUNCTIONS *= 100
CUM_PROB = np.cumsum([float(x) for x in BENEFIT_FUNCTIONS.columns])


def sim_one_car(rnd: int = 5):

    start_date = pd.Timestamp(2023, 1, 1)
    end_date = pd.Timestamp(2024, 1, 1)
    time_range = pd.date_range(start=start_date, end=end_date, freq='15min')
    day_range = pd.date_range(start=start_date, end=end_date, freq='d')

    random = np.random.default_rng(rnd)

    resident = Resident(ev_ratio=1, start_date=start_date, end_date=end_date,
                        random=random)

    car = resident.car

    hpfc = random.integers(low=0, high=999)

    prices = price_data.iloc[:, hpfc]

    if car.model == 'Nowum Car':
        return (-1, np.zeros(len(time_range)))

    col = np.argwhere(random.uniform() * 100 > CUM_PROB).flatten()
    col = col[-1] if len(col) > 0 else 0
    b_fnc = BENEFIT_FUNCTIONS.iloc[:, col]

    m = ConcreteModel()
    solver = SolverFactory('gurobi')

    segments = dict(low=[], up=[], coeff=[], low_=[])

    socs = list(b_fnc.index / 100 * car.capacity)
    benefits = list(np.cumsum(b_fnc.values * car.capacity * 0.05))

    for i in range(0, len(benefits) - 1):
        segments["low"].append(socs[i])
        segments["up"].append(socs[i + 1])
        dy = benefits[i + 1] - benefits[i]
        dx = socs[i + 1] - socs[i]
        segments["coeff"].append(dy / dx)
        segments["low_"].append(benefits[i])

    for d_time in day_range:
        t_ = range(96)
        price = prices.loc[prices.index.date == d_time.date].values

        m.clear()

        # -> declare variables
        m.ev_power = Var(t_, within=Reals, bounds=(0, None))
        m.ev_benefit = Var(t_, within=Reals, bounds=(None, None))

        m.ev_volume = Var(t_)
        m.ev_capacity_eq = ConstraintList()
        m.benefit_eq = ConstraintList()

        s = range(len(segments["low"]))

        m.z = Var(t_, s, within=Binary)
        m.q = Var(t_, s, within=Reals)

        m.choose_segment_eq = ConstraintList()
        m.s_segment_low = ConstraintList()
        m.s_segment_up = ConstraintList()

        m.ev_dbenefit_eq = ConstraintList()

        m.ev_p_dbenefit = Var(t_, within=Reals, bounds=(0, None))
        m.ev_m_dbenefit = Var(t_, within=Reals, bounds=(0, None))

        m.ev_dbenefit = Var(t_, within=Reals, bounds=(None, None))

        for t in t_:
            m.choose_segment_eq.add(quicksum(m.z[t, k] for k in s) == 1)
            for k in s:
                m.s_segment_low.add(m.q[t, k] >= segments["low"][k] * m.z[t, k])
                m.s_segment_up.add(m.q[t, k] <= segments["up"][k] * m.z[t, k])

            m.benefit_eq.add(m.ev_benefit[t] ==
                                  quicksum(segments["low_"][k] * m.z[t, k] + segments["coeff"][k]
                                           * (m.q[t, k] - segments["low"][k] * m.z[t, k])
                                           for k in s))

            m.ev_capacity_eq.add(m.ev_volume[t] == quicksum(m.q[t, k] for k in s))

            if t == 0:
                b_0 = np.interp(car.capacity * car.soc, socs, benefits)
                d_benefit = m.ev_benefit[t] - b_0
            else:
                d_benefit = m.ev_benefit[t] - m.ev_benefit[t - 1]

            m.ev_dbenefit_eq.add(m.ev_dbenefit[t] == m.ev_p_dbenefit[t] - m.ev_m_dbenefit[t])
            m.ev_dbenefit_eq.add(m.ev_dbenefit[t] == d_benefit)

        # -> limit maximal charging power
        m.ev_power_limit = ConstraintList()
        # -> limit volume to range
        m.ev_capacity_limit = ConstraintList()

        s1, s2 = d_time, d_time + td(days=1)
        usage = car.get(CarData.usage, slice(s1, s2)).values
        demand = car.get(CarData.demand, slice(s1, s2)).values

        power = car.maximal_charging_power
        volume = car.capacity * car.soc

        # -> set power and volume limits
        for t in t_:
            # -> set power limit
            m.ev_power_limit.add(m.ev_power[t] <= (1 - usage[t]) * power)
            # -> set volume limit
            m.ev_capacity_limit.add(m.ev_volume[t] >= 0)
            m.ev_capacity_limit.add(m.ev_volume[t] <= car.capacity)
            in_out = m.ev_power[t] * 0.25 - demand[t]
            if t > 0:
                volume = m.ev_volume[t - 1]
            m.ev_capacity_limit.add(m.ev_volume[t] == in_out + volume)

        ev_benefit = quicksum(m.ev_dbenefit[t] for t in t_)

        # -> declare grid variables
        m.grid_power = Var(t_, within=Reals, bounds=(0, None))

        costs = quicksum(price[t] * m.ev_power[t] * 0.25 for t in t_)

        m.obj = Objective(expr=ev_benefit - costs, sense=maximize)

        results = solver.solve(m)
        status = results.solver.status
        termination = results.solver.termination_condition

        if (status == SolverStatus.ok) and (termination == TerminationCondition.optimal):
            c_power = [value(m.ev_power[t]) for t in t_]
        else:
            c_power = [0 for _ in t_]

        car_power = pd.Series(
            data=c_power,
            index=pd.date_range(
                start=d_time, freq='15min', periods=96
            ),
        )

        car.set_final_charging(car_power)
        for t_time in time_range[time_range.date == d_time.date]:
            car.drive(t_time)
            car.charge(t_time)

    print('finished')

    return (hpfc, car.get(CarData.final_charge, time_range).values[:-1])


if __name__ == "__main__":

    col, charging = sim_one_car(rnd=8)
