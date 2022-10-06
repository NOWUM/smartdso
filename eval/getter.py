from sqlalchemy import create_engine
import pandas as pd
import numpy as np
import os

DATABASE_URI = os.getenv('DATABASE_URI', 'postgresql://opendata:opendata@10.13.10.41:5432/smartdso')
ENGINE = create_engine(DATABASE_URI)

PRICES = pd.read_csv(r'./participants/data/2022_prices.csv', index_col=0, parse_dates=True)
PRICES = PRICES.resample('15min').ffill()
PRICES['price'] /= 10


def get_typ_values(parameter: str, scenario: str, date_range: pd.DatetimeIndex = None):

    if parameter == 'market_prices':
        prices = PRICES.loc[date_range]
        data = prices.groupby(prices.index.time)['price'].mean()
        data.index = data.index.astype(str)
        return data
    elif parameter == 'charging':
        insert = "avg(final_grid) + avg(final_grid) as charging "
    else:
        insert = f"avg({parameter}) as {parameter} "

    query = f"select to_char(time, 'hh24:mi') as interval, {insert}" \
            f"from charging_summary where scenario = '{scenario}' group by interval"

    data = pd.read_sql(query, ENGINE)
    data = data.set_index('interval')

    return data


def get_sorted_values(parameter: str, scenario: str, date_range: pd.DatetimeIndex = None):
    if parameter == 'market_prices':
        prices = PRICES.loc[date_range]
        return prices.sort_values('price')
    elif parameter == 'charging':
        insert = "final_grid + final_grid as charging"
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
        insert = "sum(final_grid) + sum(final_grid) as charging"
    elif parameter == 'availability':
        insert = f"avg({parameter}) as {parameter}"
    else:
        insert = f"sum({parameter}) as {parameter}"

    query = f"select i_table.time, avg(i_table.{parameter}) as {parameter} from " \
            f"(select time, iteration, {insert} from charging_summary where scenario = '{scenario}' " \
            f"group by time, iteration order by time) i_table group by i_table.time"
    data = pd.read_sql(query, ENGINE)

    data = data.set_index('time')

    return data


def get_ev(scenario: str, ev: str):
    query = f"select time, avg(usage) as usage, avg(final_charging) as charging, " \
            f"(1-avg(usage)) * avg(pv) as used_pv_generation, avg(soc) as soc " \
            f"from electric_vehicle where id_='{ev}' and scenario='{scenario}' group by time order by time"

    data = pd.read_sql(query, ENGINE)

    data = data.set_index('time')

    return data

