import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from datetime import datetime

from eval.plotter import EvalPlotter
from eval.getter import EvalGetter

plotter = EvalPlotter()
getter = EvalGetter()

base_scenarios = ['A-PlugInCap-PV80-PriceFlat', 'A-MaxPvCap-PV80-PriceFlat',
                  'A-MaxPvCap-PV80-PriceSpot', 'A-MaxPvSoc-PV80-PriceSpot']


def get_comparison(car_id: str = 'S2C122') -> (pd.DataFrame, dict):
    compare = dict()
    sc = getter.scenarios[0]
    date_range = getter.time_ranges.get(getter.scenarios[0])
    market_price = getter.get_mean_values(scenario=sc, parameter='market_prices', date_range=date_range)
    availability = 100 * getter.get_mean_values(scenario=sc, parameter='availability').mean(axis=1)
    pv_generation = 100 * getter.get_mean_values(scenario=sc, parameter='residual_generation').mean(axis=1)
    overview = pd.concat([market_price, availability, pv_generation], axis=1)
    overview.columns = ['price', 'availability', 'pv']

    for sc in base_scenarios:
        # [scenario for scenario in getter.scenarios if 'PV25' in scenario]:
        cars = getter.cars[sc]
        car_id = next((car for car in cars if car_id in car), car_id)
        data = getter.get_ev(sc, car_id)
        idx = (data['used_pv_generation'] > 0) & (data['charging'] == 0)
        data.loc[idx, 'used_pv_generation'] = 0
        compare[sc] = data

    return overview, compare, car_id


def get_pv_impact(scenarios: list):
    impact = dict()
    for sc in scenarios:
        data = getter.get_all_utilization_values(sc)
        impact[sc] = data

    return impact


def get_pv_impact_grid_level(scenarios: list):
    transformer_impact = dict()
    line_impact = dict()
    for sc in scenarios:
        transformer_impact[sc] = getter.get_aggregated_utilization_values(sc, func='max', asset='transformer')
        line_impact[sc] = getter.get_aggregated_utilization_values(sc, func='max', asset='line')

    return transformer_impact, line_impact


def get_time_utilization(sub_id: int = 5):
    utilization = {}
    for sc in base_scenarios:
        values = []
        for iteration in range(10):
            val = getter.get_aggregated_utilization_values(scenario=sc, func='mean', asset='transformer',
                                                           iteration=iteration)
            values.append(val[sub_id].values)
        values = np.asarray(values)
        utilization[sc] = (values.max(axis=0, initial=-np.inf), values.min(axis=0, initial=np.inf), values[0, :])

    return utilization

if __name__ == "__main__":
    # summary, comparison, car_id = get_comparison()
    # fig = plotter.plot_overview(summary)
    # fig.savefig(r'./eval/plots/overview.png')
    # fig = plotter.plot_charging_compare(comparison, car_id)
    # fig.savefig(r'./eval/plots/charging_comparison.png')
    # pv_impact = get_pv_impact(scenarios=base_scenarios)
    # fig = plotter.plot_pv_impact(pv_impact)
    # fig.savefig(r'./eval/plots/pv_impact_on_transformer_scenarios.png')
    scenarios = [scenario for scenario in getter.scenarios if ('MaxPvCap' in scenario) and ('Flat' in scenario)]
    pv_impact = get_pv_impact(scenarios=base_scenarios)
    fig = plotter.plot_pv_impact(pv_impact)
    fig.savefig(r'./eval/plots/pv_impact_on_transformer_pv.png', bbox_inches = "tight")
    # https://stackoverflow.com/questions/45239261/matplotlib-savefig-text-chopped-off

    scenarios = [scenario for scenario in getter.scenarios if ('MaxPvCap' in scenario) and ('Flat' in scenario)]
    t_impact, l_impact = get_pv_impact_grid_level(scenarios=scenarios)
    fig = plotter.plot_pv_impact_grid_level(t_impact, l_impact, getter.pv_capacities)
    fig.savefig(r'./eval/plots/pv_impact_on_sub_grid.png')
    t_impact, l_impact = get_pv_impact_grid_level(scenarios=base_scenarios)
    fig = plotter.plot_pv_impact_grid_level(t_impact, l_impact, getter.pv_capacities)
    fig.savefig(r'./eval/plots/strategy_impact_on_sub_grid.png')

    # fig = plotter.plot_grid()
    # fig.savefig(r'./eval/plots/total_grid.png')
    sub_id = 5
    utilization = get_time_utilization(sub_id=sub_id)
    fig = plotter.plot_utilization(utilization, getter.time_ranges.get(getter.scenarios[0]), sub_id=sub_id)
    fig.savefig(r'./eval/plots/utilization_comparison.png')



#
# def export_summary_table() -> pd.DataFrame:
#
#     table = pd.DataFrame(columns=SCENARIOS,
#                          index=['total charging [MWh]', 'grid charging [%]', 'pv charging [%]', 'shifted charging [%]',
#                                 'cost [€/kWh]', 'market [%]', 'grid [%]',
#                                 'mean grid fee [€/kWh]', 'mean utilization [%]', '95 % quantile', 'mean > 95 % quantile'])
#
#     for SCENARIO in SCENARIOS:
#
#         initial_grid_ts = get_values(parameter='initial_grid', scenario=SCENARIO)
#         initial_grid_ts = initial_grid_ts.mean(axis=1)
#         final_grid_ts = get_values(parameter='final_grid', scenario=SCENARIO)
#         final_grid_ts = final_grid_ts.mean(axis=1)
#         final_pv_ts = get_values(parameter='final_pv', scenario=SCENARIO)
#         final_pv_ts = final_pv_ts.mean(axis=1)
#
#         initial_kwh = (initial_grid_ts.values * 0.25)
#         charged_kwh = (final_grid_ts.values * 0.25)
#         pv_kwh = (final_pv_ts.values * 0.25)
#         total_charged_kwh = charged_kwh + pv_kwh
#         idx = charged_kwh < initial_kwh
#
#         table.at['total charging [MWh]', SCENARIO] = round(total_charged_kwh.sum() / 1e3, 2)
#         table.at['grid charging [%]', SCENARIO] = round(100 * charged_kwh.sum() / total_charged_kwh.sum(), 2)
#         table.at['pv charging [%]', SCENARIO] = round(100 * pv_kwh.sum() / total_charged_kwh.sum(), 2)
#         table.at['shifted charging [%]', SCENARIO] = round(100 * (initial_kwh[idx] - charged_kwh[idx]).sum()
#                                                            / initial_kwh.sum(), 2)
#
#         market_prices = get_values(parameter='market_prices', scenario=SCENARIO, date_range=DATE_RANGE)
#         market_prices = market_prices.values.flatten()
#         if 'Flat' in SCENARIO:
#             market_cost = (charged_kwh * market_prices.mean() / 100).sum()
#         else:
#             market_cost = (charged_kwh * market_prices / 100).sum()
#
#         grid_fee = get_values(parameter='grid_fee', scenario=SCENARIO)
#         grid_fee_mean = grid_fee.mean(axis=1)
#         grid_cost = (charged_kwh * grid_fee_mean.values / 100).sum()
#
#         grid_utilization = get_grid_util(scenario=SCENARIO, func='mean')
#         grid_utilization = grid_utilization.values.mean()
#
#         total_cost = market_cost + grid_cost
#
#         table.at['cost [€/kWh]', SCENARIO] = round(total_cost / charged_kwh.sum(), 3)
#         table.at['market [%]', SCENARIO] = round(100 * market_cost / total_cost, 2)
#         table.at['grid [%]', SCENARIO] = round(100 * grid_cost / total_cost, 2)
#
#         table.at['mean grid fee [€/kWh]', SCENARIO] = round(grid_fee.values.mean() / 100, 3)
#         q95 = grid_fee.quantile(0.95)[0]
#         mean_q95 = grid_fee.values[(grid_fee > q95).values].mean()
#         table.at['95 % quantile', SCENARIO] = round(q95 / 100, 3)
#         table.at['mean > 95 % quantile', SCENARIO] = round(mean_q95 / 100, 3)
#         table.at['mean utilization [%]', SCENARIO] = round(grid_utilization, 2)
#
#     table.index.name = ''
#     table = table.rename(columns=CASE_MAPPER)
#     plot_data = summary_table(data=table.reset_index())
#     plot_data.write_image(f'./eval/plots/summary.png', width=1200, height=600)
#     plot_data.write_image(f'./eval/plots/summary.svg', width=1200, height=600)
#
#     return table
#
#
# def export_pv_synergy() -> pd.DataFrame:
#     grid_utilization = {'pv': [], 'utilization': [], 'grid': []}
#
#     for SCENARIO in PV_SCENARIOS[1:]:
#         utilization = get_grid_util(scenario=SCENARIO, func='max')
#         max_utilization = utilization.max(axis=0)
#         pv = float(PV_MAPPER[SCENARIO].split('Pv ')[-1])
#         for idx in max_utilization.index:
#             grid_utilization['grid'].append(idx + 1)
#             grid_utilization['pv'].append(pv)
#             grid_utilization['utilization'].append(max_utilization.loc[idx])
#     utilization = pd.DataFrame(grid_utilization)
#     plot_data = pv_synergy_plot(utilization)
#     width_in_mm = 90.5
#     width_default_px = 1200
#     dpi = 600
#     scale = (width_in_mm / 25.4) / (width_default_px / dpi)
#
#     plot_data.write_image(f'./eval/plots/utilization.png', scale=scale)
#     plot_data.write_image(f'./eval/plots/utilization.svg', scale=scale)
#
#     return utilization, plot_data
#
#
# def export_sorted_utilization():
#     result = {}
#
#     for SCENARIO in SCENARIOS:
#         utilization = get_sorted_values(parameter='utilization', scenario=SCENARIO)
#         print(SCENARIO, utilization.quantile(q=0.95))
#         print(SCENARIO, utilization.mean())
#         utilization = utilization.values        # [utilization.values > 1.5]
#         result[SCENARIO] = utilization.flatten()
#
#     for util in result.values():
#         x_range = 100 * np.arange(len(util))/len(util)
#         y_range = util
#         idx = (y_range > 20) & (y_range < 50)
#         plt.plot(x_range[idx], y_range[idx])
#     plt.show()
#
#     return result
