import numpy as np
import pandas as pd
from scipy.interpolate import BSpline, make_interp_spline
from collections import defaultdict

from eval.plotter import EvalPlotter
from eval.getter import EvalGetter

plotter = EvalPlotter()
getter = EvalGetter()

base_scenarios = ['F-PlugInCap-PV80-PriceFlat', 'F-MaxPvCap-PV80-PriceFlat',
                  'F-MaxPvCap-PV80-PriceSpot', 'F-MaxPvSoc-PV80-PriceSpot']
pv_scenarios = [scenario for scenario in getter.scenarios
                if ('MaxPvCap' in scenario) and ('Flat' in scenario) and ('F-' in scenario)]


def get_case(scenario: str):
    pv = scenario.split('-')[-2]
    if 'PlugInCap' in scenario:
        return f'Base Case {pv}'
    elif ('MaxPvCap' in scenario) and ('PriceFlat' in scenario):
        return f'Case A {pv}'
    elif ('MaxPvCap' in scenario) and ('PriceSpot' in scenario):
        return f'Case B {pv}'
    elif ('MaxPvSoc' in scenario) and ('PriceSpot' in scenario):
        return f'Case C {pv}'


def smooth_function(x: np.array, y: np.array):
    new_x = np.linspace(x.min(), x.max(), 100)
    spl = make_interp_spline(x, y, k=3)
    new_y = spl(new_x)
    return new_x, new_y


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


def get_impact(scenarios: list):
    impact = {sc: getter.get_all_utilization_values(sc) for sc in scenarios}
    return impact


def get_impact_grid_level(scenarios: list):
    transformer_impact = {sc: getter.get_aggregated_utilization_values(sc, func='max', asset='transformer')
                          for sc in scenarios}
    total_charged = {sc: getter.get_total_charged_per_grid(sc) for sc in scenarios}

    return transformer_impact, total_charged


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


def get_economic_table(scenarios: list):
    table = pd.DataFrame(columns=scenarios,
                         index=['total charging [MWh]', 'grid charging [%]', 'pv charging [%]', 'shifted charging [%]',
                                'cost [€/kWh]', 'market [%]', 'grid [%]',
                                'mean grid fee [€/kWh]', 'grid fee 95 % quantile', 'grid fee mean > 95 % quantile',
                                'mean utilization [%]', 'utilization 95 % quantile', 'utilization mean > 95 % quantile'])

    for sc in scenarios:
        # return value for each simulation run
        initial_grid_ts = getter.get_mean_values(parameter='initial_grid', scenario=sc)
        # initial_grid_ts = initial_grid_ts.mean(axis=1)
        # return value for each simulation run
        final_grid_ts = getter.get_mean_values(parameter='final_grid', scenario=sc)
        # final_grid_ts = final_grid_ts.mean(axis=1)
        # return value for each simulation run
        final_pv_ts = getter.get_mean_values(parameter='final_pv', scenario=sc)
        # final_pv_ts = final_pv_ts.mean(axis=1)
        # return value for each simulation run
        grid_fee = getter.get_mean_values(parameter='grid_fee', scenario=sc)

        sim_range = getter.time_ranges.get(getter.scenarios[0])
        market_prices = getter.get_mean_values(parameter='market_prices', scenario=sc, date_range=sim_range)
        market_prices = market_prices.values.flatten()

        r = defaultdict(list)
        for col in initial_grid_ts.columns:
            initial_kwh = (initial_grid_ts[col].values * 0.25)
            charged_kwh = (final_grid_ts[col].values * 0.25)
            pv_kwh = (final_pv_ts[col].values * 0.25)
            total_charged_kwh = charged_kwh + pv_kwh
            shifted = np.abs(charged_kwh - initial_kwh).sum() / initial_kwh.sum()

            r['shifted'].append(shifted)
            r['total_charged'].append(total_charged_kwh.sum())
            r['grid_charged'].append(charged_kwh.sum()/total_charged_kwh.sum())
            r['pv_charged'].append(pv_kwh.sum()/total_charged_kwh.sum())

            if 'Flat' in sc:
                market_cost = (charged_kwh * market_prices.mean() / 100).sum()
            else:
                market_cost = (charged_kwh * market_prices / 100).sum()

            grid_cost = (charged_kwh * grid_fee[col].values / 100).sum()
            total_cost = market_cost + grid_cost

            r['total_cost'].append(total_cost/total_charged_kwh.sum())
            r['grid_cost'].append(grid_cost/total_cost)
            r['market_cost'].append(market_cost/total_cost)

        table.at['total charging [MWh]', sc] = round(np.mean(r['total_charged']) / 1e3, 2)
        table.at['grid charging [%]', sc] = round(100 * np.mean(r['grid_charged']), 0)
        table.at['pv charging [%]', sc] = round(100 * np.mean(r['pv_charged']), 0)
        table.at['shifted charging [%]', sc] = round(100 * np.mean(r['shifted']), 1)

        table.at['cost [€/kWh]', sc] = np.mean(r['total_cost'])
        table.at['market [%]', sc] = np.mean(r['market_cost'])
        table.at['grid [%]', sc] = np.mean(r['grid_cost'])

        table.at['mean grid fee [€/kWh]', sc] = grid_fee.values.mean().mean()
        q95 = grid_fee.quantile(0.95)[0]
        table.at['grid fee 95 % quantile', sc] = q95
        mean_q95 = grid_fee.values[(grid_fee > q95).values].mean()
        table.at['grid fee mean > 95 % quantile', sc] = mean_q95

        grid_utilization = getter.get_all_utilization_values(scenario=sc)
        mean_utilization = grid_utilization.values.mean()
        table.at['mean utilization [%]', sc] = round(mean_utilization, 0)

        q95 = grid_utilization.quantile(0.95)[0]
        mean_q95 = grid_utilization.values[(grid_utilization > q95).values].mean()
        table.at['utilization 95 % quantile', sc] = round(q95, 0)
        table.at['utilization mean > 95 % quantile', sc] = round(mean_q95, 0)

    return table


def export_strategy_impact_on_transformer_level():
    strategy_impact = get_impact(scenarios=base_scenarios)
    strategy_impact = {get_case(sc): val for sc, val in strategy_impact.items()}
    fig = plotter.plot_impact(strategy_impact)
    # https://stackoverflow.com/questions/45239261/matplotlib-savefig-text-chopped-off
    fig.savefig(r'./eval/plots/strategy_impact_on_transformer_pv.svg', bbox_inches="tight")


def export_economic_table():
    table = get_economic_table(base_scenarios)
    table.to_excel('./eval/plots/results.xlsx')


def export_benefit_function():
    benefit_functions = getter.get_benefit_functions()
    x_values = benefit_functions.index.values
    idx = [x in [0, 15, 40, 80, 100] for x in x_values]
    x_values = x_values[idx]
    new_y_values = {}
    for col in benefit_functions.columns:
        y_values = benefit_functions[col].values
        y_values = y_values[idx]
        _, y_values = smooth_function(x_values, y_values)
        new_y_values[col] = y_values
    benefit_functions = pd.DataFrame(new_y_values)
    fig = plotter.plot_benefit_function(benefit_functions)
    fig.savefig(r'./eval/plots/benefit_functions.svg')


def export_grid_fee_function():
    grid_fee = getter.get_grid_fee()
    fig = plotter.plot_grid_fee_function(grid_fee)
    fig.savefig(r'./eval/plots/grid_fee_function.svg')


def export_charging_compare():
    _, comparison, car_id = get_comparison()
    comparison = {get_case(sc): val for sc, val in comparison.items()}
    fig = plotter.plot_charging_compare(comparison, car_id)
    fig.savefig(r'./eval/plots/charging_comparison.svg')


def export_market_pv_availability():
    summary, _, car_id = get_comparison()
    fig = plotter.plot_overview(summary)
    fig.savefig(r'./eval/plots/overview.svg')


def export_pv_impact_on_transformer_level():
    pv_impact = get_impact(scenarios=pv_scenarios)
    pv_impact = {get_case(sc): val for sc, val in pv_impact.items()}
    fig = plotter.plot_impact(pv_impact, pv_colors=True)
    # https://stackoverflow.com/questions/45239261/matplotlib-savefig-text-chopped-off
    fig.savefig(r'./eval/plots/pv_impact_on_transformer.svg', bbox_inches="tight")


def export_pv_impact_on_grid_level():
    t_impact, total_charged = get_impact_grid_level(scenarios=pv_scenarios)
    t_impact = {get_case(sc): val for sc, val in t_impact.items()}
    total_charged = {get_case(sc): val for sc, val in total_charged.items()}
    fig = plotter.plot_pv_impact_grid_level(t_impact, total_charged, getter.pv_capacities)
    fig.savefig(r'./eval/plots/pv_impact_on_sub_grid.svg')


def export_strategy_impact_on_grid_level():
    t_impact, total_charged = get_impact_grid_level(scenarios=base_scenarios)
    t_impact = {get_case(sc): val for sc, val in t_impact.items()}
    total_charged = {get_case(sc): val for sc, val in total_charged.items()}
    fig = plotter.plot_impact_grid_level(t_impact, total_charged, getter.pv_capacities)
    fig.savefig(r'./eval/plots/strategy_impact_on_sub_grid.svg')


if __name__ == "__main__":

    export_charging_compare()
    export_market_pv_availability()
    export_economic_table()
    export_strategy_impact_on_transformer_level()
    export_pv_impact_on_transformer_level()
    export_benefit_function()
    export_grid_fee_function()
    export_pv_impact_on_grid_level()
    # export_strategy_impact_on_grid_level()
    # #
    # t_impact, total_charged = get_impact_grid_level(scenarios=base_scenarios)
    # t_impact = {get_case(sc): val for sc, val in t_impact.items()}
    # total_charged = {get_case(sc): val for sc, val in total_charged.items()}
    #
    # fig = plotter.plot_impact_grid_level(t_impact, total_charged, getter.pv_capacities)
    # fig.savefig(r'./eval/plots/strategy_impact_on_sub_grid.png')
    #
    # # fig = plotter.plot_grid()
    # # fig.savefig(r'./eval/plots/total_grid.png')
    # sub_id = 5
    # utilization = get_time_utilization(sub_id=sub_id)
    # utilization = {get_case(sc): val for sc, val in utilization.items()}
    # fig = plotter.plot_utilization(utilization, getter.time_ranges.get(getter.scenarios[0]), sub_id=sub_id)
    # fig.savefig(r'./eval/plots/utilization_comparison.png')

