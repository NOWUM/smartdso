import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sqlalchemy import create_engine

DATABASE_URI = os.getenv('DATABASE_URI', 'postgresql://opendata:opendata@10.13.10.41:5432/mobsim')


class EnergyProvider:

    def __init__(self, table: str = 'charging'):

        self.engine = create_engine(DATABASE_URI)
        self.table = table

    def get_time_series(self, num: int = 200):
        query = f"select time, iteration, power from {self.table} where iteration <= {num}"
        df = pd.read_sql(query, self.engine)
        df = df.set_index(['time', 'iteration'])
        return df


if __name__ == "__main__":

    omega_size = 600

    ep = EnergyProvider()
    data = ep.get_time_series(omega_size)
    data = np.asarray([data.loc[:, i, :].values.flatten() for i in range(1, omega_size)], dtype=float).T

    vars = [(np.var(data[:, list(range(k))], axis=0)).mean() for k in range(1, 50)]
    plt.plot(vars)
    plt.show()

    vars = {}
    portfolio_size = 50
    sample_size = 100
    for k in range(400, 500):
        sample = []
        for n in range(sample_size):
            idx = list(np.random.choice(range(k), portfolio_size))
            sample.append(data[:, idx].mean(axis=1))
        sample = np.asarray(sample, dtype=float)
        var = np.var(sample, axis=0).mean()
        vars[k] = var

    plt.plot(vars.keys(), vars.values())
    plt.show()