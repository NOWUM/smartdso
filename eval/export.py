import numpy as np
import pandas as pd
import os
from datetime import timedelta as td

from eval.getter import get_typ_values, get_sorted_values, get_values
from matplotlib import pyplot as plt
from eval.plotter import overview

start_date = pd.to_datetime(os.getenv('START_DATE', '2022-05-01'))              # -> default start date
end_date = pd.to_datetime(os.getenv('END_DATE', '2022-05-31'))                  # -> default end date
DATE_RANGE = pd.date_range(start=start_date, end=end_date + td(hours=23, minutes=45), freq='15min')
WEEK_RANGE = pd.date_range(start=start_date + td(days=3),
                           end=start_date + td(days=10, hours=23, minutes=45),
                           freq='15min')

SCENARIO = 'EV100PV25PRCFlatSTRPlugInCap'

if __name__ == "__main__":

    market_price = get_values(scenario=SCENARIO, parameter='market_prices', date_range=DATE_RANGE)
    market_price.columns = ['market_price']
    availability = get_values(scenario=SCENARIO, parameter='availability')
    availability *= 100
    pv_generation = get_values(scenario=SCENARIO, parameter='residual_generation')
    pv_generation /= pv_generation.values.max()
    pv_generation *= 100
    pv_generation.columns = ['pv_generation']

    data = pd.concat([market_price, availability, pv_generation], axis=1)
    overview(data=data.loc[WEEK_RANGE]).write_image(f'./eval/plots/overview.svg', width=1200, height=600)

    # -> get/build typical days
    # charging = get_typ_values(scenario=scenario, parameter='charging')
    # market_prices = get_typ_values(scenario=scenario, parameter='market_prices', date_range=DATE_RANGE)
    # availability = get_typ_values(scenario=scenario, parameter='availability')
    # grid_fees = get_typ_values(scenario=scenario, parameter='grid_fee')
    # pv_generation = get_typ_values(scenario=scenario, parameter='residual_generation')
    #
    # # -> sorted grid fees
    # sorted_grid_fees = get_sorted_values(scenario=scenario, parameter='grid_fee')
