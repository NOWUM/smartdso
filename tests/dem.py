from demLib.electric_profile import StandardLoadProfile
import pandas as pd
from matplotlib import pyplot as plt

DEMAND_DATA = pd.read_csv(r'./gridLib/data/export/dem/consumers.csv', index_col=0)
DATE_RANGE = pd.date_range(start='2022-01-01', periods=365, freq='d')
profile_types = {'H0': 'household', 'G0': 'business', 'RLM': 'industry'}


def test_profile(t: str):
    data = DEMAND_DATA.loc[DEMAND_DATA['profile'] == t]  # -> all h0 df
    print(f'Total Demand in CSV: {round(data["jeb"].sum() / 1e3, 2)} MWh')
    model = StandardLoadProfile(demandP=data["jeb"].sum(), type=profile_types[t])
    plt_data = model.run_model(pd.Timestamp(2022, 5, 1))
    plt.plot(plt_data)
    plt.title(t)
    plt.show()
    demand = 0
    for day in DATE_RANGE:
        d = model.run_model(day)
        demand += 0.25 * d.sum()
    print(f'Sum SLP Profile: {round(demand / 1e3, 2)} MWh')
    delta = 100 * ((demand - data["jeb"].sum()) / data["jeb"].sum())
    print(f'Delta: {round(delta, 2)} %')
    return delta


if __name__ == "__main__":
    for t_ in ['H0', 'G0', 'RLM']:
        print(f'checking {t_}')
        d = test_profile(t_)
        assert d > 5
        print('----------------------------')