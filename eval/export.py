import numpy as np
import pandas as pd
import os
from datetime import timedelta as td

from eval.getter import (get_typ_values, get_sorted_values, 
                        get_values, get_ev,get_total_values, 
                        get_grid, get_avg_soc, get_shifted,
                        get_grid_avg_sub, get_gzf_count, get_gzf_power,
                        get_cars, get_scenarios, get_soc)

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

SCENARIOS = ['A-PlugInCap-PV25-PriceFlat', 'A-MaxPvCap-PV25-PriceFlat',
             'A-MaxPvCap-PV80-PriceSpot', 'A-MaxPvSoc-PV80-PriceSpot']
SCENARIO = SCENARIOS[0]

def export_comparison():

    compare = dict()

    market_price = get_values(scenario=SCENARIOS[0], parameter='market_prices', date_range=DATE_RANGE)
    #market_price = market_price.mean(axis=1)

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
        for car in cars:
            if 'S2C122' in car:
                break
        

        data = get_ev(SCENARIO, car)
        plot_data = ev_plot(data=data.loc[WEEK_RANGE])
        plot_data.write_image(f'./eval/plots/single/ev_{SCENARIO}.svg', width=1200, height=600)
        data = data.loc[WEEK_RANGE]
        index = (data['used_pv_generation'] > 0) & (data['charging'] == 0)
        data.loc[index, 'used_pv_generation'] = 0
        compare[SCENARIO] = data

    plot_data = scenario_compare(data=compare, num_rows=len(compare))
    plot_data.write_image(f'./eval/plots/compare.svg', width=1200, height=600)


if __name__ == "__main__":

    export_comparison()

    results = dict()

    table = {key:{} for key in SCENARIOS}

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
        table[SCENARIO]['total_cost']=total_cost
        table[SCENARIO]['total_cost_kwh']=total_cost/charged_kwh.sum()
        table[SCENARIO]['total_cost_residential']=total_cost / 3185
        table[SCENARIO]['charged_sum'] = charged_kwh.sum()

        # -> sorted grid fees
        sorted_grid_fees = get_sorted_values(scenario=SCENARIO, parameter='grid_fee')
    
        qmean = sorted_grid_fees.mean()[0]
        q5= sorted_grid_fees.quantile(0.05)[0]
        q95 = sorted_grid_fees.quantile(0.95)[0]
        qmin = sorted_grid_fees.min()[0]
        qmax = sorted_grid_fees.max()[0]

        table[SCENARIO]['grid_fee_q5'] = q5
        table[SCENARIO]['grid_fee_mean'] = qmean
        table[SCENARIO]['grid_fee_q95'] = q95
        plt.plot(sorted_grid_fees)

        table[SCENARIO]['charging_kWh'] = get_total_values(parameter='charging', scenario=SCENARIOS[0])
        table[SCENARIO]['charging_grid_kWh'] = get_total_values(parameter='final_grid', scenario=SCENARIOS[0])
        table[SCENARIO]['charging_pv_kWh'] = get_total_values(parameter='final_pv', scenario=SCENARIOS[0])
        table[SCENARIO]['initial_grid_kWh'] = get_total_values(parameter='initial_grid', scenario=SCENARIOS[0])
        table[SCENARIO]['distance_km'] = get_total_values(parameter='distance', scenario=SCENARIOS[0])

        # Verschoben aufgrund Marktsignal (diff initial - final)
        table[SCENARIO]['shifted'] = get_shifted(SCENARIO).mean()

        # Gleichzeitigkeitsfaktor
        table[SCENARIO]['gzf'] = get_gzf_power(SCENARIO)['gzf'].mean()
        table[SCENARIO]['gzf_count'] = get_gzf_count(SCENARIO)['gzf'].mean()
        grid_data = get_grid_avg_sub(scenario=SCENARIO)
        j = 0
        for i in grid_data.mean():
            print(i, j)
            table[SCENARIO][f'avg_grid_util_sub{j}'] = i
            j +=1

        result = []

        results[SCENARIO] = get_grid_avg_sub(scenario=SCENARIO)

    for value in results.values():
        value.mean(axis=1).plot()
        plt.show()

    ################# TABLE #########

    df = pd.DataFrame(table)
    market_prices = get_values(parameter='market_prices', scenario='MaxPvCap-PV80-PriceSpot', date_range=DATE_RANGE)
    scenario = 'A-MaxPvCap-PV80-PriceSpot'
    value = results[scenario]
    plt.scatter(market_prices.values, value.mean(axis=1), label=scenario)
    # plt.show()
    scenario = 'A-MaxPvSoc-PV80-PriceSpot'
    value = results[scenario]
    plt.scatter(market_prices.values, value.mean(axis=1), label=scenario)
    plt.legend()
    plt.xlabel('ct/kWh')
    plt.ylabel('grid util %')
    plt.show()

    # charging = data['charging'].values[0] * 0.25 / 1e3

    #     #plot_data.show()
    #
        
    # # -> get/build typical days
    #
    # scenario = SCENARIO
    # charging = get_typ_values(scenario=scenario, parameter='charging')
    # market_prices = get_typ_values(scenario=scenario, parameter='market_prices', date_range=DATE_RANGE)
    # availability = get_typ_values(scenario=scenario, parameter='availability')
    # grid_fees = get_typ_values(scenario=scenario, parameter='grid_fee')
    # pv_generation = get_typ_values(scenario=scenario, parameter='residual_generation')
