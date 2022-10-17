from sqlalchemy import create_engine
import pandas as pd
import numpy as np
import os

DATABASE_URI = os.getenv('DATABASE_URI', 'postgresql://opendata:opendata@10.13.10.41:5432/smartdso')
ENGINE = create_engine(DATABASE_URI)

PRICES = pd.read_csv(r'./participants/data/2022_prices.csv', index_col=0, parse_dates=True)
PRICES = PRICES.resample('15min').ffill()
PRICES['price'] /= 10

def get_scenarios():
    query = 'select distinct scenario from electric_vehicle where time=(select time from electric_vehicle order by time limit 1) and iteration = 0 order by scenario'
    data = pd.read_sql(query, ENGINE)
    return list(data['scenario'])

def get_cars(scenario):
    query = f"select distinct id_ from electric_vehicle where scenario='{scenario}' and time=(select time from electric_vehicle order by time limit 1) order by id_"
    data = pd.read_sql(query, ENGINE)
    return list(data['id_'])

def get_soc(scenario: str, sort='desc'):
    query = f"select id_, soc from electric_vehicle where scenario='{scenario}' and time=(select time from electric_vehicle order by time {sort} limit 1)"
    data = pd.read_sql(query, ENGINE)
    return data

def get_avg_soc(scenario: str):
    query = f"select time, avg(soc) as avg_soc from electric_vehicle where scenario='{scenario}' group by time order by time"
    data = pd.read_sql(query, ENGINE, index_col = 'time')
    return data

def get_auslastung(scenario: str, asset:str = 'line'):
    query = f"select time, avg(utilization), id_ from grid_asset where asset='{asset}' and scenario='{scenario}' group by time order by time desc limit 1"
    data = pd.read_sql(query, ENGINE)
    return data

def get_typ_values(parameter: str, scenario: str, date_range: pd.DatetimeIndex = None):

    if parameter == 'market_prices':
        prices = PRICES.loc[date_range]
        data = prices.groupby(prices.index.time)['price'].mean()
        data.index = data.index.astype(str)
        return data
    elif parameter == 'charging':
        insert = "avg(final_grid) + avg(final_pv) as charging "
    else:
        insert = f"avg({parameter}) as {parameter} "

    query = f"select to_char(time, 'hh24:mi') as interval, {insert}" \
            f"from charging_summary where scenario = '{scenario}' group by interval"

    data = pd.read_sql(query, ENGINE, index_col='interval')
    return data


def get_sorted_values(parameter: str, scenario: str, date_range: pd.DatetimeIndex = None):
    if parameter == 'market_prices':
        prices = PRICES.loc[date_range]
        return prices.sort_values('price')
    elif parameter == 'charging':
        insert = "final_grid + final_pv as charging"
    else:
        insert = f"{parameter} as {parameter}"

    query = f"select {insert} from charging_summary where scenario = '{scenario}' order by {parameter} desc"
    data = pd.read_sql(query, ENGINE)

    return data

def get_values(parameter: str, scenario: str, date_range: pd.DatetimeIndex = None):
    if parameter == 'market_prices':
        prices = PRICES.loc[date_range]
        return prices
    elif parameter == 'charging':
        insert = "sum(final_grid) + sum(final_pv) as charging"
    elif parameter in ['availability', 'grid_fee']:
        insert = f"avg({parameter}) as {parameter}"
    else:
        insert = f"sum({parameter}) as {parameter}"

    query = f"""select i_table.time, iteration, avg(i_table.{parameter}) as {parameter} from (
                    select time, iteration, {insert} from charging_summary where scenario = '{scenario}'
                    group by time, iteration order by time
                    ) i_table group by i_table.time, i_table.iteration
            """
    data = pd.read_sql(query, ENGINE, index_col='time')
    return data.pivot(columns='iteration', values=parameter)


def get_ev(scenario: str, ev: str):
    query = f"select time, avg(usage) as usage, avg(final_charging) as charging, " \
            f"(1-avg(usage)) * avg(pv) as used_pv_generation, avg(soc) as soc " \
            f"from electric_vehicle where id_='{ev}' and scenario='{scenario}' group by time order by time"

    data = pd.read_sql(query, ENGINE, index_col='time')
    return data


def get_total_values(parameter: str, scenario: str):
    table = 'charging_summary'
    factor = 1
    insert = f"sum({parameter}) as {parameter}"
    if parameter in ['charging', 'final_grid', 'final_pv', 'initial_grid']:
        if parameter == 'charging':
            insert = "sum(final_grid) + sum(final_pv) as charging"
        factor = 0.25 / 1e3
    elif parameter == 'availability':
        insert = f"avg({parameter}) as {parameter}"
    elif parameter == 'distance':
        table = 'electric_vehicle'

    query = f"select iteration, {insert} from {table} where scenario='{scenario}' " \
            f"group by iteration"

    data = pd.read_sql(query, ENGINE, index_col='iteration')
    data = data.mean() * factor

    return data.values[0]


def get_shifted(scenario: str):

    query = f"""select iteration, sum(initial_grid - final_grid)
        from charging_summary
        where initial_grid > final_grid and scenario='{scenario}'
        group by iteration
        """
    data = pd.read_sql(query, ENGINE, index_col='iteration')

    return data

def get_grid(scenario: str, iteration: int, sub_id=None, func='max'):
    '''average utilization for a given iteration'''
    if sub_id:
        query = f"select time, avg(value) as util_{iteration} from grid_summary where scenario = '{scenario}' and type='{func}' " \
            f"and iteration={iteration} group by time order by time"
    else:
        query = f"select time, avg(value) as util_{iteration} from grid_summary where scenario = '{scenario}' and type='{func}' " \
                f"and iteration={iteration} group by time order by time"

    data = pd.read_sql(query, ENGINE, index_col='time')
    return data


def get_grid_util(scenario: str, iteration: int=None, func='max'):
    """average utilization through all sub grids"""
    if func == 'max':
        name = func
        func = name
    else:
        name = func
        func = 'avg'

    if iteration is None:
        query = f"""select time, sub_id, {func}(value) as util 
                from grid_summary where scenario = '{scenario}' and type='{name}'
                group by sub_id, time order by time
                """
    else:
        query = f"""select time, sub_id, {func}(value) as util_{iteration} from grid_summary 
                where scenario = '{scenario}' and type='{name}' and iteration={iteration}
                group by sub_id, time order by time
                """

    data = pd.read_sql(query, ENGINE, index_col='time')
    return data.pivot(columns='sub_id', values='util')

def pv_capacity():
    total_alloc = pd.read_csv(fr'./gridLib/data/grid_allocations.csv', index_col=0)
    tc = total_alloc.dropna()
    tc = tc['pv']
    summ_pdc_alloc = 0
    for val in tc:
        l = eval(val)
        for i in l:
            summ_pdc_alloc += i['pdc0']

    total_consumers = pd.read_csv(fr'./gridLib/data/export/dem/consumers.csv', index_col=0)
    tc = total_consumers.dropna()
    tc = tc['pv']
    summ_pdc_consumer = 0
    for val in tc:
        l = eval(val)
        for i in l:
            summ_pdc_consumer += i['pdc0']
    return summ_pdc_alloc, summ_pdc_consumer # [kWp]

def get_max_power(scenario: str) -> float:
    query = f"select id_, max(final_charging) as max_power from electric_vehicle where scenario = '{scenario}' group by id_"
    data = pd.read_sql(query, ENGINE, index_col='id_')
    return float(data.sum()) # [kW]

def get_gzf(scenario: str, typ='power', typ_tage = False):
    if typ_tage:
        typ_agg = "to_char(time, 'hh24:mi')"
        q = f"select count(*) from (select distinct time from charging_summary where scenario = '{scenario}') a"
        days = int(pd.read_sql(q, ENGINE)['count'])/96
    else:
        typ_agg = 'time'
        days = 1

    if typ =='power':
        query = f"""
            select {typ_agg} as interval, sum(final_grid + final_pv) as charging
            from charging_summary where scenario = '{scenario}'
            group by interval order by interval
            """
    elif typ=='count':
        query = f"""select {typ_agg} as interval, count(CASE WHEN final_charging > 0 THEN 1 END) as gzf
            from electric_vehicle
            where scenario = '{scenario}' group by interval order by interval
            """
    data = pd.read_sql(query, ENGINE, index_col='interval')

    if typ =='power':
        data['gzf'] = data['charging']/get_max_power(scenario)
    else:
        count = len(get_cars(scenario))
        data = data/count
    return data/days

if __name__ == '__main__':
    sc = get_scenarios()
    scenario = sc[0]
    df_last = get_soc(scenario)
    # histogram of last charging states
    # df_last.hist()
    df_last['soc'].mean()
    df_first = get_soc(scenario, 'asc')
    df_first['soc'].mean()
    #df_first.hist()

    # curves of average soc per scenario
    import matplotlib.pyplot as plt
    for i in range(4):
        d = get_avg_soc(sc[i])
        plt.plot(d, label = sc[i])
    plt.legend()
    plt.show()


    for i in range(10):
        utilization = get_grid(scenario, iteration=i)
        overloaded_hours = len(utilization[utilization[f'util_{i}'] > 60])/4
    utilization.plot()
    utilization2 = get_grid2(scenario, iteration=1)

    shifted = get_shifted(scenario)

    gzf = get_gzf(scenario, typ='power')
    gzf_count = get_gzf(scenario, typ='count')
    gzf['gzf'].plot()
    gzf_count['gzf'].plot()

    # dauerlastkurve der auslastung:
    kurve = gzf_count.sort_values('gzf')['gzf']
    kurve.plot()

    typ_gzf = get_gzf(scenario, typ='power', typ_tage=True)
    typ_gzf_count = get_gzf(scenario, typ='count', typ_tage=True)

    grid = get_grid_avg_sub(scenario, func='mean')
    util = grid.mean(axis=1)
    charged = get_values('charging', scenario)
    import matplotlib.pyplot as plt
    # TODO in plotly graph überführen
    plt.scatter(util, charged['charging'], label=scenario)
    plt.legend()
    plt.xlabel('mittlere Netzauslastung %')
    plt.ylabel('kW Ladeleistung')
    charged['charging']
    grid.sum()