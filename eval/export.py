import pandas as pd

from eval.getter import get_mean_charging, get_prices
from eval.plotter import plot_mean_charging

if __name__ == "__main__":
    cases = dict(A=['EV100PV50PRC40.0STR-S', 'EV100PV80PRC40.0STR-S', 'EV100PV100PRC40.0STR-S'],
                 B=['EV100PV50PRC40.0STR-SPV', 'EV100PV80PRC40.0STR-SPV', 'EV100PV100PRC40.0STR-SPV'])

    for case, scenarios in cases.items():
        result = []
        for scenario in scenarios:
            data, energy = get_mean_charging(scenario)
            data.columns = [scenario]
            result.append(data.copy())
        result = pd.concat(result, axis=1)
        figure = plot_mean_charging(result)
        figure.write_html(f'case_{case}.html')

