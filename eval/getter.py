from sqlalchemy import create_engine
import pandas as pd
import numpy as np
import os

DATABASE_URI = os.getenv('DATABASE_URI', 'postgresql://opendata:opendata@10.13.10.41:5432/smartdso')
ENGINE = create_engine(DATABASE_URI)


def get_prices():
    prices = pd.read_csv(r'./participants/data/2022_prices.csv', index_col=0, parse_dates=True)
    prices = prices.resample('15min').ffill()
    prices['price'] /= 10
    return prices


def get_price_sensitivities(slopes: tuple = (1.0, 1.7, 2.7, 4.0), max_price: float = 40.0):
    senses = []
    for slope in slopes:
        X = 100/slope
        sens = [max_price * np.exp(-x/X)/X for x in np.arange(0.1, 100.1, 0.1)]
        senses += [np.asarray(sens)]
    return senses


def get_mean_charging(scenario):
    query = f"select to_char(car.ti, 'hh24:mi') as inter, avg(car.final)/1000 as charging " \
            f"from (select  time as ti, sum(final_charge) as final " \
            f"from cars where scenario='{scenario}' group by ti) as car " \
            f"group by inter"
    data = pd.read_sql(query, ENGINE)
    data = data.set_index('inter')

    query = f"select avg(result.charging) as mean_charged " \
            f"from (select iteration, 0.25 * sum(final_charge)/1000 as charging " \
            f"from cars where scenario='{scenario}' group by iteration) as result"

    mean_charged = pd.read_sql(query, ENGINE)
    mean_charged = mean_charged.values[0][0]

    return data, mean_charged
