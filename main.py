import pandas as pd
from tqdm import tqdm
import logging
import os
from matplotlib import pyplot as plt

from agents.flexibility_provider import FlexibilityProvider
from agents.capacity_provider import CapacityProvider
from utils import TableCreator

logging.basicConfig()

logger = logging.getLogger('Simulation')
logger.setLevel('DEBUG')

# -> timescaledb connection to store the simulation results
DATABASE_URI = os.getenv('DATABASE_URI', 'postgresql://opendata:opendata@10.13.10.41:5432/smartgrid')

try:
    tables = TableCreator(create_tables=False, database_uri=DATABASE_URI)
    logger.info(' -> connected to database')
except Exception as e:
    logger.error(f" -> can't connect to {DATABASE_URI}")
    logger.error(repr(e))
    engine = None
    raise Exception("Bad simulation parameters, please check your input")

start_date = pd.to_datetime(os.getenv('START_DATE', '2015-08-01'))              # -> default start date
end_date = pd.to_datetime(os.getenv('END_DATE', '2015-08-10'))                  # -> default end date

logger.info(f' -> initialize simulation for {start_date.date()} - {end_date.date()}')

scenario_name = os.getenv('SCENARIO_NAME', 'TEST100_0')
sim = os.getenv('RESULT_PATH', scenario_name.split('_')[-1])

logger.info(f' -> scenario {scenario_name.split("_")[0]}')

save_demand_as_csv = (os.getenv('SAVE_DEMAND', 'False') == 'True')
plot = False

input_set = {'london_data': (os.getenv('LONDON_DATA', 'True') == 'True'),      # -> Need london data set
             'start_date': start_date,                                         #    see: demLib.london_data.py
             'end_date': end_date,
             'T': int(os.getenv('STEPS_PER_DAY', 96)),
             'ev_ratio': int(os.getenv('EV_RATIO', 100))/100,
             'pv_ratio': int(os.getenv('PV_RATIO', 80))/100,
             'scenario': scenario_name.split('_')[0],
             'iteration': sim,
             'database_uri': DATABASE_URI}

try:
    FlexProvider = FlexibilityProvider(**input_set)
    logger.info(' -> started Flexibility Provider')
    logging.getLogger('FlexibilityProvider').setLevel('WARNING')
    CapProvider = CapacityProvider(**input_set, write_geo=False)
    logger.info(' -> started Capacity Provider')
    logging.getLogger('CapacityProvider').setLevel('WARNING')
except Exception as e:
    logger.error(f" -> can't initialize agents")
    logger.error(repr(e))
    raise Exception("Bad simulation parameters, please check your input")


if __name__ == "__main__":

    try:
        # -> run SLPs for each day in simulation horizon
        logger.info(f' -> running photovoltaic and slp generation')
        fixed_demand, fixed_generation = FlexProvider.initialize_time_series()
        # -> forward the slp data to the Capacity Provider
        logger.info(f' -> running initial power flow calculation')
        CapProvider.set_fixed_power(data=fixed_demand)
        if save_demand_as_csv:
            if input_set.get('london_data'):
                fixed_demand.groupby('t').sum().to_csv('London.csv', sep=';', decimal=',')
            else:
                fixed_demand.groupby('t').sum().to_csv('SLP.csv', sep=';', decimal=',')
        if plot:
            fixed_demand.groupby('t').sum().plot()
            fixed_generation.groupby('t').sum().plot()
            plt.show()
    except Exception as e:
        print(repr(e))
        logger.error(f' -> error while slp or power flow calculation: {repr(e)}')

    # -> start simulation for date range start_date till end_date
    logger.info(' -> starting simulation')
    for day in pd.date_range(start=start_date, end=end_date, freq='d'):
        logger.info(f' -> running day {day.date()}')
        logger.info(' -> consumers plan charging...')
        try:
            for d_time in tqdm(pd.date_range(start=day, periods=96, freq='15min')):
                while FlexProvider.get_commits() < len(FlexProvider.keys):
                    for request, node_id, consumer_id in FlexProvider.get_requests(d_time=d_time):
                        price = CapProvider.get_price(request=request, node_id=node_id)
                        if FlexProvider.commit(price, consumer_id):
                            CapProvider.commit(request=request, node_id=node_id)
                    logger.info(f' -> {FlexProvider.get_commits()} consumers commit charging')
                FlexProvider.simulate(d_time)
            FlexProvider.save_results(day)
            CapProvider.save_results(day)
        except Exception as e:
            logger.error(f' -> error during simulation: {repr(e)}')

