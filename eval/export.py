import numpy as np
import pandas as pd
import os
from datetime import timedelta as td
from collections import defaultdict

from eval.getter import (get_typ_values, get_sorted_values,
                         get_values, get_ev, get_total_values,
                         get_grid, get_avg_soc, get_shifted,
                         get_grid_util, get_gzf_count, get_gzf_power,
                         get_cars, get_scenarios, get_soc)

from matplotlib import pyplot as plt
from eval.plotter import overview, ev_plot, scenario_compare, summary_table, pv_synergy_plot

start_date = pd.to_datetime(os.getenv('START_DATE', '2022-05-01'))  # -> default start date
end_date = pd.to_datetime(os.getenv('END_DATE', '2022-05-31'))  # -> default end date
DATE_RANGE = pd.date_range(start=start_date, end=end_date + td(hours=23, minutes=45), freq='15min')
DAY_OFFSET = 14
WEEK_RANGE = pd.date_range(start=start_date + td(days=DAY_OFFSET),
                           end=start_date + td(days=DAY_OFFSET + 7, hours=23, minutes=45),
                           freq='15min')

# 'MaxPvCap-PV50-PriceFlat', 'MaxPvCap-PV80-PriceFlat', 'MaxPvCap-PV100-PriceFlat'

SCENARIOS = ['A-PlugInCap-PV25-PriceFlat', 'A-MaxPvCap-PV25-PriceFlat',
             'A-MaxPvCap-PV80-PriceSpot', 'A-MaxPvSoc-PV80-PriceSpot']

SCENARIO = SCENARIOS[0]

CASE_MAPPER = {'A-PlugInCap-PV25-PriceFlat': 'Base Case',
               'A-MaxPvCap-PV25-PriceFlat': 'Case A',
               'A-MaxPvCap-PV80-PriceSpot': 'Case B',
               'A-MaxPvSoc-PV80-PriceSpot': 'Case C'}

PV_SCENARIOS = ['A-PlugInCap-PV25-PriceFlat', 'A-MaxPvCap-PV25-PriceFlat',
                'A-MaxPvCap-PV50-PriceFlat',  'A-MaxPvCap-PV80-PriceFlat', 'A-MaxPvCap-PV100-PriceFlat']

PV_MAPPER = {'A-PlugInCap-PV25-PriceFlat': 'Base Case Pv 25',
             'A-MaxPvCap-PV25-PriceFlat': 'Case A Pv 25',
             'A-MaxPvCap-PV50-PriceFlat': 'Case A Pv 50',
             'A-MaxPvCap-PV80-PriceFlat': 'Case A Pv 80',
             'A-MaxPvCap-PV100-PriceFlat': 'Case A Pv 100'}


def export_comparison(car_id: str = 'S2C122') -> dict:
    compare = dict()

    market_price = get_values(scenario=SCENARIOS[0], parameter='market_prices', date_range=DATE_RANGE)

    availability = get_values(scenario=SCENARIOS[0], parameter='availability')
    availability = availability.mean(axis=1)
    availability *= 100

    pv_generation = get_values(scenario=SCENARIOS[0], parameter='residual_generation')
    pv_generation = pv_generation.mean(axis=1)
    pv_generation /= pv_generation.values.max()
    pv_generation *= 100

    data = pd.concat([market_price, availability, pv_generation], axis=1)
    data.columns = ['market_price', 'availability', 'pv_generation']
    plot_data = overview(data=data.loc[WEEK_RANGE])
    plot_data.write_image(f'./eval/plots/single/overview.svg', width=1200, height=600)

    compare['overview'] = data.loc[WEEK_RANGE].resample('h').mean()

    for SCENARIO in SCENARIOS:
        grid_fees = get_sorted_values(scenario=SCENARIO, parameter='grid_fee')
        plt.semilogx(grid_fees, label=SCENARIO)
        plt.legend()
        cars = get_cars(SCENARIO)
        car_id = next((car for car in cars if car_id in car), car_id)

        data = get_ev(SCENARIO, car_id)
        plot_data = ev_plot(data=data.loc[WEEK_RANGE])
        plot_data.write_image(f'./eval/plots/single/ev_{SCENARIO}.svg', width=1200, height=600)
        data = data.loc[WEEK_RANGE]
        idx = (data['used_pv_generation'] > 0) & (data['charging'] == 0)
        data.loc[idx, 'used_pv_generation'] = 0
        compare[SCENARIO] = data

    plot_data = scenario_compare(data=compare, num_rows=len(compare))
    plot_data.write_image(f'./eval/plots/compare.svg', width=1200, height=600)

    return compare


def export_summary_table() -> pd.DataFrame:

    table = pd.DataFrame(columns=SCENARIOS,
                         index=['total charging [MWh]', 'grid charging [%]', 'pv charging [%]', 'shifted charging [%]',
                                'cost [€/kWh]', 'market [%]', 'grid [%]',
                                'mean grid fee [€/kWh]', 'mean utilization [%]', '95 % quantile', 'mean > 95 % quantile'])

    for SCENARIO in SCENARIOS:

        initial_grid_ts = get_values(parameter='initial_grid', scenario=SCENARIO)
        initial_grid_ts = initial_grid_ts.mean(axis=1)
        final_grid_ts = get_values(parameter='final_grid', scenario=SCENARIO)
        final_grid_ts = final_grid_ts.mean(axis=1)
        final_pv_ts = get_values(parameter='final_pv', scenario=SCENARIO)
        final_pv_ts = final_pv_ts.mean(axis=1)

        initial_kwh = (initial_grid_ts.values * 0.25)
        charged_kwh = (final_grid_ts.values * 0.25)
        pv_kwh = (final_pv_ts.values * 0.25)
        total_charged_kwh = charged_kwh + pv_kwh
        idx = charged_kwh < initial_kwh

        table.at['total charging [MWh]', SCENARIO] = round(total_charged_kwh.sum() / 1e3, 2)
        table.at['grid charging [%]', SCENARIO] = round(100 * charged_kwh.sum() / total_charged_kwh.sum(), 2)
        table.at['pv charging [%]', SCENARIO] = round(100 * pv_kwh.sum() / total_charged_kwh.sum(), 2)
        table.at['shifted charging [%]', SCENARIO] = round(100 * (initial_kwh[idx] - charged_kwh[idx]).sum()
                                                           / initial_kwh.sum(), 2)

        market_prices = get_values(parameter='market_prices', scenario=SCENARIO, date_range=DATE_RANGE)
        market_prices = market_prices.values.flatten()
        if 'Flat' in SCENARIO:
            market_cost = (charged_kwh * market_prices.mean() / 100).sum()
        else:
            market_cost = (charged_kwh * market_prices / 100).sum()

        grid_fee = get_values(parameter='grid_fee', scenario=SCENARIO)
        grid_fee_mean = grid_fee.mean(axis=1)
        grid_cost = (charged_kwh * grid_fee_mean.values / 100).sum()

        grid_utilization = get_grid_util(scenario=SCENARIO, func='mean')
        grid_utilization = grid_utilization.values.mean()

        total_cost = market_cost + grid_cost

        table.at['cost [€/kWh]', SCENARIO] = round(total_cost / charged_kwh.sum(), 3)
        table.at['market [%]', SCENARIO] = round(100 * market_cost / total_cost, 2)
        table.at['grid [%]', SCENARIO] = round(100 * grid_cost / total_cost, 2)

        table.at['mean grid fee [€/kWh]', SCENARIO] = round(grid_fee.values.mean() / 100, 3)
        q95 = grid_fee.quantile(0.95)[0]
        mean_q95 = grid_fee.values[(grid_fee > q95).values].mean()
        table.at['95 % quantile', SCENARIO] = round(q95 / 100, 3)
        table.at['mean > 95 % quantile', SCENARIO] = round(mean_q95 / 100, 3)
        table.at['mean utilization [%]', SCENARIO] = round(grid_utilization, 2)

    table.index.name = ''
    table = table.rename(columns=CASE_MAPPER)
    plot_data = summary_table(data=table.reset_index())
    plot_data.write_image(f'./eval/plots/summary.svg', width=1200, height=600)

    return table


def export_pv_synergy() -> pd.DataFrame:
    grid_utilization = {'pv': [], 'utilization': [], 'grid': []}

    for SCENARIO in PV_SCENARIOS[1:]:
        utilization = get_grid_util(scenario=SCENARIO, func='max')
        max_utilization = utilization.max(axis=0)
        pv = float(PV_MAPPER[SCENARIO].split('Pv ')[-1])
        for idx in max_utilization.index:
            grid_utilization['grid'].append(idx + 1)
            grid_utilization['pv'].append(pv)
            grid_utilization['utilization'].append(max_utilization.loc[idx])

    return pd.DataFrame(grid_utilization)


if __name__ == "__main__":
    # car charging comparison
    # overview = export_comparison()
    # summary_table = export_summary_table()
    utilization = export_pv_synergy()
    plot_data = pv_synergy_plot(utilization)
    plot_data.write_image(f'./eval/plots/utilization.svg', width=1200, height=600)