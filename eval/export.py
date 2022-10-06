import numpy as np
import pandas as pd
import os
from datetime import timedelta as td

from eval.getter import get_typ_values, get_sorted_values, get_values, get_ev
from matplotlib import pyplot as plt
from eval.plotter import overview, ev_plot, scenario_compare

start_date = pd.to_datetime(os.getenv('START_DATE', '2022-05-01'))              # -> default start date
end_date = pd.to_datetime(os.getenv('END_DATE', '2022-05-31'))                  # -> default end date
DATE_RANGE = pd.date_range(start=start_date, end=end_date + td(hours=23, minutes=45), freq='15min')
DAY_OFFSET = 14
WEEK_RANGE = pd.date_range(start=start_date + td(days=DAY_OFFSET),
                           end=start_date + td(days=DAY_OFFSET + 7, hours=23, minutes=45),
                           freq='15min')


# 'MaxPvCap-PV50-PriceFlat', 'MaxPvCap-PV80-PriceFlat', 'MaxPvCap-PV100-PriceFlat'

SCENARIOS = ['PlugInCap-PV25-PriceFlat-L', 'MaxPvCap-PV25-PriceFlat-L',
             'MaxPvCap-PV80-PriceSpot-L', 'MaxPvSoc-PV80-PriceSpot-L']


if __name__ == "__main__":

    compare = dict()

    SCENARIO = SCENARIOS[0]

    market_price = get_values(scenario=SCENARIO, parameter='market_prices', date_range=DATE_RANGE)

    market_price.columns = ['market_price']
    availability = get_values(scenario=SCENARIO, parameter='availability')
    availability *= 100

    pv_generation = get_values(scenario=SCENARIO, parameter='residual_generation')
    pv_generation /= pv_generation.values.max()
    pv_generation *= 100
    pv_generation.columns = ['pv_generation']

    data = pd.concat([market_price, availability, pv_generation], axis=1)
    plot_data = overview(data=data.loc[WEEK_RANGE])
    plot_data.write_image(f'./eval/plots/single/overview.svg', width=1200, height=600)

    compare['overview'] = data.loc[WEEK_RANGE].resample('h').mean()

    for SCENARIO in SCENARIOS:

        data = get_ev(SCENARIO, 'S2C123')
        plot_data = ev_plot(data=data.loc[WEEK_RANGE])
        plot_data.write_image(f'./eval/plots/single/ev_{SCENARIO}.svg', width=1200, height=600)
        data = data.loc[WEEK_RANGE]
        index = (data['used_pv_generation'] > 0) & (data['charging'] == 0)
        data.loc[index, 'used_pv_generation'] = 0
        compare[SCENARIO] = data

    plot_data = scenario_compare(data=compare, num_rows=len(compare))
    plot_data.write_image(f'./eval/plots/compare.svg', width=1200, height=600)
    #     #plot_data.show()
    #
    #     # -> sorted grid fees
    #     sorted_grid_fees = get_sorted_values(scenario=SCENARIO, parameter='grid_fee')
    #
    #     qmean = sorted_grid_fees.mean()[0]
    #     q5= sorted_grid_fees.quantile(0.05)[0]
    #     q95 = sorted_grid_fees.quantile(0.95)[0]
    #     qmin = sorted_grid_fees.min()[0]
    #     qmax = sorted_grid_fees.max()[0]
    #     print(f'Mean {qmean:.3}, Q05 {q5:.3}, Q95 {q95:.3}, Max {qmax:.3}, Min {qmin:.3}')
    #     plt.plot(sorted_grid_fees)
    # # -> get/build typical days
    #
    # scenario = SCENARIO
    # charging = get_typ_values(scenario=scenario, parameter='charging')
    # market_prices = get_typ_values(scenario=scenario, parameter='market_prices', date_range=DATE_RANGE)
    # availability = get_typ_values(scenario=scenario, parameter='availability')
    # grid_fees = get_typ_values(scenario=scenario, parameter='grid_fee')
    # pv_generation = get_typ_values(scenario=scenario, parameter='residual_generation')