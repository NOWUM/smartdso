import numpy as np
import pandas as pd
import os


from eval.getter import get_typ_values, get_sorted_values
from eval.plotter import plot_mean_charging
from matplotlib import pyplot as plt

start_date = pd.to_datetime(os.getenv('START_DATE', '2022-05-01'))              # -> default start date
end_date = pd.to_datetime(os.getenv('END_DATE', '2022-05-31'))                  # -> default end date
DATE_RANGE = pd.date_range(start=start_date, end=end_date, freq='15min')

if __name__ == "__main__":
    scenario = 'EV100PV25PRCFlatSTRPlugInCap'
    # -> get/build typical days
    charging = get_typ_values(scenario=scenario, parameter='charging')
    market_prices = get_typ_values(scenario=scenario, parameter='market_prices', date_range=DATE_RANGE)
    availability = get_typ_values(scenario=scenario, parameter='availability')
    grid_fees = get_typ_values(scenario=scenario, parameter='grid_fee')
    pv_generation = get_typ_values(scenario=scenario, parameter='residual_generation')

    # -> sorted grid fees
    sorted_grid_fees = get_sorted_values(scenario=scenario, parameter='grid_fee')
