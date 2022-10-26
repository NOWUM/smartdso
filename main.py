import pandas as pd
from tqdm import tqdm
import logging
import os

from agents.flexibility_provider import FlexibilityProvider
from agents.capacity_provider import CapacityProvider
from utils import TableCreator

logging.basicConfig()
logger = logging.getLogger('Simulation')
logger.setLevel('INFO')

# -> timescaledb connection to store the simulation results
DATABASE_URI = os.getenv('DATABASE_URI', 'postgresql://opendata:opendata@10.13.10.41:5432/smartdso')

try:
    tables = TableCreator(create_tables=False, database_uri=DATABASE_URI)
    logger.info(' -> connected to database')
except Exception as e:
    logger.error(f" -> can't connect to {DATABASE_URI}")
    logger.error(repr(e))
    engine = None
    raise Exception("Bad simulation parameters, please check your input")

start_date = pd.to_datetime(os.getenv('START_DATE', '2022-05-01'))              # -> default start date
end_date = pd.to_datetime(os.getenv('END_DATE', '2022-05-10'))                  # -> default end date

logger.info(f' -> initialize simulation for {start_date.date()} - {end_date.date()}')

scenario_name = os.getenv('SCENARIO_NAME', 'Test_0')
# -> PlugInCap, MaxPvCap, MaxPvSoc, PlugInInf
strategy = os.getenv('STRATEGY', 'PlugInCap')
sim = int(os.getenv('RESULT_PATH', scenario_name.split('_')[-1]))

if 'Test' in scenario_name:
    tables.delete_scenario(scenario=scenario_name.split('_')[0])

logger.info(f' -> scenario {scenario_name.split("_")[0]} and iteration {sim}')
analyse_grid = os.getenv('ANALYSE_GRID', 'True') == 'True'

input_set = {'london_data': (os.getenv('LONDON_DATA', 'False') == 'True'),      # -> Need london data set
             'start_date': start_date,                                          #    see: demLib.london_data.py
             'end_date': end_date,
             'T': int(os.getenv('STEPS_PER_DAY', 96)),
             'ev_ratio': int(os.getenv('EV_RATIO', 100))/100,
             'pv_ratio': int(os.getenv('PV_RATIO', 100))/100,
             'number_consumers': int(os.getenv('NUMBER_CONSUMERS', 0)),
             'scenario': scenario_name.split('_')[0],
             'iteration': sim,
             'strategy': strategy,
             'sub_grid': int(os.getenv('SUB_GRID', 5)),
             'database_uri': DATABASE_URI}


try:
    logger.info(' -> starting Capacity Provider')
    CapProvider = CapacityProvider(**input_set, write_geo=False)
    logger.info(' -> started Capacity Provider')
    logging.getLogger('CapacityProvider').setLevel('DEBUG')
    logger.info(' -> starting Flexibility Provider')
    FlexProvider = FlexibilityProvider(grid_series=CapProvider.mapper, **input_set)
    logger.info(' -> started Flexibility Provider')
    logging.getLogger('FlexibilityProvider').setLevel('DEBUG')
except Exception as e:
    logger.error(f" -> can't initialize agents")
    logger.error(repr(e))
    raise Exception(f"Bad simulation parameters, please check your input {repr(e)}")


if __name__ == "__main__":

    try:
        # -> run SLPs for each day in simulation horizon
        logger.info(f' -> running photovoltaic and slp generation')
        fixed_demand, fixed_generation = FlexProvider.initialize_time_series()
        if analyse_grid:
            # -> forward the slp data to the Capacity Provider
            logger.info(f' -> running initial power flow calculation')
            CapProvider.set_fixed_power(data=fixed_demand)
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
                number_commits = 0
                while number_commits < len(FlexProvider.keys):
                    for request, node_id, consumer_id in FlexProvider.get_requests(d_time=d_time):
                        if analyse_grid:
                            price = CapProvider.get_price(request=request, node_id=node_id)
                            if FlexProvider.commit(price, consumer_id):
                                CapProvider.commit(request=request, node_id=node_id)
                        else:
                            price = pd.Series(index=request.index, data=[0] * len(request))
                            FlexProvider.commit(price, consumer_id)

                    if 'MaxPv' in strategy:
                        number_commits = FlexProvider.get_commits()
                        logger.debug(f' -> {FlexProvider.get_commits()} consumers commit charging')
                    elif 'PlugIn' in strategy:
                        logger.debug('set commit charging for clients')
                        number_commits = len(FlexProvider.keys)
                FlexProvider.simulate(d_time)
            FlexProvider.save_results(day)
            if analyse_grid:
                CapProvider.save_results(day)
                pass

        except Exception as e:
            logger.exception(f'Error during simulation')

