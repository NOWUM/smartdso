import numpy as np
import pandas as pd
import os
from datetime import timedelta as td

from eval.getter import get_typ_values, get_sorted_values, get_values, get_ev,get_total_values, get_grid
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


def export_comparison():

    compare = dict()

    market_price = get_values(scenario=SCENARIOS[0], parameter='market_prices', date_range=DATE_RANGE)

    market_price.columns = ['market_price']
    availability = get_values(scenario=SCENARIOS[0], parameter='availability')
    availability *= 100

    pv_generation = get_values(scenario=SCENARIOS[0], parameter='residual_generation')
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


if __name__ == "__main__":

    # export_comparison()

    charging = get_total_values(parameter='charging', scenario=SCENARIOS[0])
    final_grid = get_total_values(parameter='final_grid', scenario=SCENARIOS[0])
    final_pv = get_total_values(parameter='final_pv', scenario=SCENARIOS[0])
    initial_grid = get_total_values(parameter='initial_grid', scenario=SCENARIOS[0])
    distance = get_total_values(parameter='distance', scenario=SCENARIOS[0])

    results = dict()

    for SCENARIO in SCENARIOS:
        initial_grid_ts = get_values(parameter='initial_grid', scenario=SCENARIO)
        final_grid_ts = get_values(parameter='final_grid', scenario=SCENARIO)
        initial_kwh = (initial_grid_ts.values * 0.25)
        charged_kwh = (final_grid_ts.values * 0.25)
        index = charged_kwh < initial_kwh
        shifted = 100 * (initial_kwh[index] - charged_kwh[index]).sum()/initial_kwh.sum()
        print(shifted)

        market_prices = get_values(parameter='market_prices', scenario=SCENARIO, date_range=DATE_RANGE)
        if 'Flat' in SCENARIO:
            market_cost = (charged_kwh * market_prices.values.mean() / 100).sum()
        else:
            market_cost = (charged_kwh * market_prices.values / 100).sum()

        grid_fee = get_values(parameter='grid_fee', scenario=SCENARIO)
        grid_cost = (charged_kwh * grid_fee.values / 100).sum()

        total_cost = market_cost + grid_cost
        total_cost_kwh = total_cost/charged_kwh.sum()
        total_cost_residential = total_cost / 3185

        print(SCENARIO, total_cost, total_cost_kwh, total_cost_residential, charged_kwh.sum())

        result = []

        for i in range(10):
            data = get_grid(scenario=SCENARIO, iteration=i)
            result.append(data)
        results[SCENARIO] = pd.concat(result, axis=1)

    for value in results.values():
        value.mean(axis=1).plot()
        plt.show()
    market_prices = get_values(parameter='market_prices', scenario='MaxPvCap-PV80-PriceSpot-L', date_range=DATE_RANGE)
    value = results['MaxPvCap-PV80-PriceSpot-L']
    plt.scatter(market_prices.values, value.mean(axis=1))
    # plt.show()
    value = results['MaxPvSoc-PV80-PriceSpot-L']
    plt.scatter(market_prices.values, value.mean(axis=1))
    plt.show()

    # charging = data['charging'].values[0] * 0.25 / 1e3

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