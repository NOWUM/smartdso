import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from pyomo.environ import Constraint, Var, Objective, SolverFactory, ConcreteModel, \
     Reals, Binary, minimize, value, quicksum, ConstraintList


def optimize(portfolio_size: int = 50, samples: int = 5_000):
    print(' -> starting energy provider')
    ep = EnergyProvider()
    print(f' -> set portfolio size to {portfolio_size} with {samples} samples')
    ep.set_portfolio(portfolio_size, samples)
    print(' -> starting optimization')
    model = ep.optimize(r_interest=0.1)
    print(f' -> result for portfolio size {portfolio_size}:  {round(value(model.obj),2)} €/MWh')


class EnergyProvider:

    def __init__(self, year: int = 2023):

        self.charging_data = np.load(r'./pricingLib/charging_data.pkl')
        self.ts = len(self.charging_data[0])
        self.price_data = pd.read_pickle(r'./pricingLib/price_data.pkl')
        self.price_data = self.price_data.loc[self.price_data.index.year == year]

        self._tm_price = np.asarray(list(self.price_data.values)).mean()

        self.portfolio_size = 50
        self.samples = 30

        self.beta = 0.95
        self.r_interest = 0.1

        # -> optimization input & output
        self._portfolios = []
        self._prices = []
        self._losses = []
        self._model = ConcreteModel()
        self._volume_da = []

    def set_portfolio(self, size: int = 50, samples: int = 30,
                      one_price: bool = False, one_volume: bool = False):

        self.portfolio_size = size
        self._portfolios = []
        self.samples = samples
        self._prices = []

        if one_volume:
            vol = np.random.randint(low=0, high=self.charging_data.shape[0])
            portfolios = [size * [vol] for _ in range(samples)]
        else:
            portfolios = [np.random.choice(range(self.charging_data.shape[0]), size)
                          for _ in range(samples)]

        for portfolio in portfolios:
            charging_ts = self.charging_data[portfolio].mean(axis=0)
            self._portfolios.append(charging_ts)
            if one_price:
                price = self.price_data.values.mean(axis=1) / 10
            else:
                price = self.price_data.values[:, np.random.randint(low=0, high=999)] / 10
            self._prices.append(price)

        self._tm_price = np.asarray(self._prices).mean()

    def optimize(self, beta: float = 0.95, r_interest: float = 0.2, price_fixed: bool = True):

        self.beta = beta
        self.r_interest = r_interest

        m = ConcreteModel()
        solver = SolverFactory('gurobi')

        samples = [i for i in range(self.samples)]
        ts = [i for i in range(self.ts)]
        # -> consumer tariff (fix price for consumer)
        m.tariff = Var(bounds=(0, None), within=Reals)
        # -> volume @ future market set upper bound to max power in portfolio
        max_power = round(np.asarray(self._portfolios).max(initial=0), 2)
        m.volume_tm = Var(bounds=(0, max_power), within=Reals)
        # -> value at risk (VaR)
        m.value_at_risk = Var(bounds=(0, None), within=Reals)
        # -> vars to linearize the cVar function
        m.z = Var(samples, bounds=(0, None), within=Reals)

        # -> cost @ future market
        cost_tm = 0.25 * quicksum(m.volume_tm for _ in ts) * self._tm_price

        # -> volume @ day ahead market
        volume_da = np.asarray(self._portfolios) - m.volume_tm
        # -> cost @ day ahead market
        cost_da = [0.25 * quicksum(volume_da[s][t] * self._prices[s][t] for t in ts)
                   for s in samples]

        # -> income from consumer side
        if price_fixed:
            # -> calculate revenue
            revenue = [0.25 * quicksum(self._portfolios[s][t] * m.tariff for t in ts)
                       for s in samples]
        else:
            # -> calculate revenue
            revenue = [0.25 * quicksum(volume_da[s][t] * self._prices[s][t]
                                       + m.volume_tm * cost_tm + m.tariff * self._portfolios[s][t]
                                       for t in ts)
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

        solver.solve(m)

        self._losses = [value(l) for l in loss]
        self._volume_da = volume_da

        self._model = m

    def evaluate_result(self):
        print('{:#^50}'.format(' Risk Parameters '))
        print(f' -> Tariff: {round(value(self._model.obj),2)} ct/kWh')
        # -> get VaR and cVaR
        sort_loss = np.sort(self._losses)
        VaR = sort_loss[int(self.beta * len(sort_loss))]
        cVaR = np.mean(sort_loss[int(self.beta * len(sort_loss)):])
        print(f' -> VaR: {round(VaR/100,2)} €')
        print(f' -> cVaR: {round(cVaR/100,2)} €')

        print('{:#^50}'.format(' Volumes '))
        print(f' -> Volume Future Market: {round(value(self._model.volume_tm),2)}')

        # -> plot histogram of loss
        f, ax = plt.subplots(1, 1)
        ax.hist(self._losses, bins=max(int(self.samples/10), 5))
        ax.grid(True)
        ax.set_xlabel('loss in [€]')
        ax.set_ylabel('frequency [#]')
        ax.axvline(VaR, color='k', linestyle='dashed', linewidth=1)
        ax.axvline(cVaR, color='r', linestyle='dashed', linewidth=1)

        plt.show()


if __name__ == "__main__":

    ep = EnergyProvider()
    ep.set_portfolio(100, 1000, one_price=True)
    ep.optimize(r_interest=0.1, price_fixed=True)
    ep.evaluate_result()
    # m1 = ep.optimize(r_interest=0.1, price_fixed=False)



