import os
import pickle
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sqlalchemy import create_engine

from pyomo.environ import Constraint, Var, Objective, SolverFactory, ConcreteModel, \
     Reals, Binary, minimize, value, quicksum, ConstraintList


class EnergyProvider:

    def __init__(self):

        self.charging_data = pickle.load(open(r'./agents/data/charging_data.pkl', 'rb'))
        self.ts = len(self.charging_data[0])
        self.price_data = pickle.load(open(r'./agents/data/price_data.pkl', 'rb'))

        self._tm_price = np.asarray(list(self.price_data.values())).mean() / 1e3
        self._da_price = np.asarray(list(self.price_data.values())).mean(axis=0) / 1e3

        self.portfolio_size = 50
        self.samples = 30

        self._portfolios = []
        self._prices = []

        self._losses = []

        self.results = {'losses': [],
                        'tariff': []}

    def set_portfolio(self, size: int = 50, samples: int = 30):
        self.portfolio_size = size
        self._portfolios = []
        self.samples = samples
        self._prices = []

        portfolios = [np.random.choice(list(self.charging_data.keys()), size) for _ in range(samples)]

        for portfolio in portfolios:
            charging_ts = np.asarray([self.charging_data[num] for num in portfolio])
            charging_ts = charging_ts.mean(axis=0)
            self._portfolios.append(charging_ts)
            price = self.price_data[portfolio[0]][:int(self.ts/4)]
            self._prices.append(np.repeat(price, 4))

        self._tm_price = np.asarray(self._prices).mean()
        self._da_price = np.asarray(list(self.price_data.values())).mean(axis=0)

    def optimize(self, beta: float = 0.95, r_interest: float = 0.2, price_fixed: bool = True):
        m = ConcreteModel()
        solver = SolverFactory('gurobi')

        samples = [i for i in range(self.samples)]
        timesteps = [i for i in range(self.ts)]
        # -> consumer tariff (fix price for consumer)
        m.tariff = Var(bounds=(0, None), within=Reals)
        # -> volume @ future market (without upper boundary)
        m.volume_tm = Var(bounds=(0, None), within=Reals)
        # -> value at risk (VaR)
        m.value_at_risk = Var(bounds=(0, None), within=Reals)
        # -> vars to linearize the cVar function
        m.z = Var(samples, bounds=(0, None), within=Reals)

        # -> volume @ day ahead market
        volume_da = np.asarray(self._portfolios) - m.volume_tm

        revenue = [0.25 * quicksum(self._portfolios[s][t] * m.tariff for t in timesteps)
                   for s in samples]

        # -> cost @ future market
        cost_tm = 0.25 * quicksum(m.volume_tm for _ in timesteps) * self._tm_price
        # -> income from consumer side
        if price_fixed:
            # -> cost @ day ahead market
            cost_da = [0.25 * quicksum(volume_da[s][t] * self._prices[s][t] for t in timesteps)
                       for s in samples]

        else:
            price = np.asarray(self._prices).mean(axis=0)
            cost_da = [0.25 * quicksum(volume_da[s][t] * (self._prices[s][t] - price[t])
                                       for t in timesteps)
                       for s in samples]

        # -> loss in each scenario
        loss = [cost_da[s] + cost_tm - revenue[s] for s in samples]
        # -> mean loss over all samples (negative = win)
        avg_loss = 1 / self.samples * quicksum(loss[s] for s in samples)
        # -> conditional value at risk
        c_value_at_risk = m.value_at_risk + (1 / ((1 - beta) * self.samples) * quicksum(m.z[s] for s in samples))
        # -> indifference constraint
        m.risk_eq = Constraint(expr=avg_loss + r_interest * c_value_at_risk == 0)

        # -> constraint z's
        m.z_eq = ConstraintList()

        for s in samples:
            m.z_eq.add(m.z[s] >= loss[s] - m.value_at_risk)

        m.obj = Objective(expr=m.tariff, sense=minimize)

        solve = solver.solve(m)

        self.results['losses'] = [value(loss[s]) for s in samples]

        self._losses = [value(l) for l in loss]

        print(value(m.obj))

        return m


if __name__ == "__main__":

    from matplotlib import pyplot as plt
    from matplotlib.ticker import PercentFormatter
    from tqdm import tqdm
    fig, ax = plt.subplots(5, 1, tight_layout=True)
    # kwargs = dict(histtype='stepfilled', alpha=1.0, bins=100, range=(0, 8))
    ep = EnergyProvider()
    ep.set_portfolio(30, 100)
    ep.optimize(r_interest=0.0)
    plt.hist(ep._losses, bins=30)
    ep.optimize(r_interest=0.1)
    plt.hist(ep._losses, bins=30)

    plt.show()



    # mu = {size: [] for size in (30, 50, 100, 200, 300, 500)}
    # sigma = {size: [] for size in (30, 50, 100, 200, 300, 500)}
    # for _ in tqdm(range(30)):
    #     for size in (30, 50, 100, 200, 300, 500):
    #         ep.set_portfolio(30, 250)
    #         ep.optimize(price_fixed=True)
    #         mu[size].append(np.mean(ep.results['losses']))
    #         sigma[size].append((np.var(ep.results['losses']))**(1/2))

    # for k, row in zip([10, 100, 200, 500, 1000], range(5)):
    #     print(k)
    #     ep.set_portfolio(k, 200)
    #     x = np.asarray(ep._portfolios)
    #     n, bins, patches = ax[row].hist(x.flatten(), **kwargs)
    #     ax[row].yaxis.set_major_formatter(PercentFormatter(xmax=len(x.flatten() / n.max())))
    #     ax[row].legend([k])
    #     ax[row].grid(True)
    #     ax[row].set_ylim([0, len(x.flatten()) * 0.20])
    #
    # plt.show()
    # r1 = ep.optimize(price_fixed=False)
    #
    # plt.hist(ep.results['losses'], bins=30)
    # plt.show()
    #
    # r2 = ep.optimize()
    #
    # plt.hist(ep.results['losses'], bins=30)
    # plt.show()
    #
    # tariff = value(r2.obj)
    # plt.plot(np.repeat([tariff], ep.ts))
    #
    # tariff = value(r2.obj)
    # plt.plot(np.repeat([tariff], ep.ts))
    #
    # plt.show()

    # for k in range(10):
    #     ep.set_portfolio(50, 150)
    #     ep.build_model()