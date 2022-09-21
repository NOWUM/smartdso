import numpy as np
import pandas as pd

from eval.getter import get_mean_charging, get_prices, get_mean_pv_consumption, get_mean_charging_iteration, get_charging_iteration
from eval.plotter import plot_mean_charging
from matplotlib import pyplot as plt


if __name__ == "__main__":
    scenarios = ['EV100PV50PRCFlatSTRPlugInCap', 'EV100PV80PRCFlatSTRPlugInCap',
                 'EV100PV100PRCFlatSTRPlugInCap']
    scenario = 'EV100PV100PRCFlatSTRPlugInInf'

    for size in range(50, 250):
        samples = [np.random.choice(range(15000), size) for i in range(500)]

        result = dict()
        for sample, index in zip(samples, range(3)):
            power = get_charging_iteration(scenario=scenario, iteration=tuple(sample))
            result[index] = power.values.flatten()

        # Var berechnen

        for r in result.values():
            plt.plot(r)
        plt.show()
    # result = dict()
    # for i in range(0, 1325, 25):
    #     print(i)
    #     power = get_mean_charging_iteration(scenario=scenario, num=i)
    #
    #     first_deviation = np.abs(np.diff(power.values.flatten()))
    #     second_deviation = np.abs(np.diff(first_deviation))
    #     result[i] = np.sum(second_deviation)
    #
    # plt.plot(result.keys(), result.values())
    #
    # values = []
    # for scenario in scenarios:
    #     data = get_mean_pv_consumption(scenario)
    #     values.append(data.values)
    #
    # to_store = {}
    # for scenario, value in zip(scenarios, values):
    #     to_store[scenario] = value.flatten()
    # to_store = pd.DataFrame(data=to_store, index=data.index)
    #
    # typ_days = {scenario: [] for scenario in scenarios}
    # for scenario in scenarios:
    #     values = []
    #     for day in range(7):
    #         day_data = to_store.loc[to_store.index.day_of_week == day, scenario]
    #         value = day_data.groupby(day_data.index.hour).mean().values.flatten()
    #         values += list(value)
    #     typ_days[scenario] = values
    #
    # typ_days = pd.DataFrame(data=typ_days)