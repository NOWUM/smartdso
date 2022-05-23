import pandas as pd
from tqdm import tqdm
import logging
import os
from datetime import timedelta as td
from sqlalchemy import create_engine, inspect

from agents.flexibility_provider import FlexibilityProvider
from agents.capacity_provider import CapacityProvider

logging.basicConfig()

logger = logging.getLogger('Simulation')
logger.setLevel('INFO')

start_date = pd.to_datetime(os.getenv('START_DATE', '2022-01-01'))              # -> default start date
end_date = pd.to_datetime(os.getenv('END_DATE', '2022-01-15'))                  # -> default end date

logger.info(f' -> initialize simulation for {start_date.date()} - {end_date.date()}')

scenario_name = os.getenv('SCENARIO_NAME', 'EV100LIMIT-1DFTRUE_0')
sim = os.getenv('RESULT_PATH', scenario_name.split('_')[-1])

logger.info(f' -> scenario {scenario_name.split("_")[0]}')

save_demand_as_csv = (os.getenv('SAVE_DEMAND', 'False') == 'True')

input_set = {'london_data': (os.getenv('LONDON_DATA', 'False') == 'True'),      # -> Need london data set
             'dynamic_fee': (os.getenv('DYNAMIC_FEE', 'True') == 'True'),       #    see: demLib.london_data.py
             'minimum_soc': int(os.getenv('MINIMUM_SOC', -1)),
             'start_date': start_date,
             'end_date': end_date,
             'ev_ratio': int(os.getenv('EV_RATIO', 100))/100,
             'base_price': int(os.getenv('BASE_PRICE', 29)),
             'scenario': scenario_name.split('_')[0],
             'iteration': sim}

try:
    FlexProvider = FlexibilityProvider(**input_set)
    logger.info(' -> started Flexibility Provider')
    logging.getLogger('FlexibilityProvider').setLevel('WARNING')
    CapProvider = CapacityProvider(**input_set)
    logger.info(' -> started Capacity Provider')
    logging.getLogger('CapacityProvider').setLevel('WARNING')
except Exception as e:
    logger.error(f" -> can't initialize agents")
    logger.error(repr(e))
    raise Exception("Bad simulation parameters, please check your input")


# -> timescaledb connection to store the simulation results
DATABASE_URI = 'postgresql://opendata:opendata@10.13.10.41:5432/smartdso'
try:
    engine = create_engine(DATABASE_URI)
    tables = inspect(engine).get_table_names()
    logger.info(' -> connected to database')
except Exception as e:
    logger.error(f" -> can't connect to {DATABASE_URI}")
    logger.error(repr(e))
    engine = None
    raise Exception("Bad simulation parameters, please check your input")

if __name__ == "__main__":

    try:
        # -> run SLPs for each day in simulation horizon
        logger.info(f' -> running slp - generation for {start_date.date()} - {end_date.date()}')
        fixed_power = []
        for day in tqdm(pd.date_range(start=start_date, end=end_date, freq='d')):
            fixed_power += [FlexProvider.get_fixed_power(day)]
        # -> forward the slp data to the Capacity Provider
        logger.info(f' -> running power flow calculation for {len(CapProvider.mapper.unique())} grids')
        CapProvider.set_fixed_power(data=pd.concat(fixed_power))
        if save_demand_as_csv:
            if input_set.get('london_data'):
                pd.concat(fixed_power).groupby('t').sum().to_csv('London.csv', sep=';', decimal=',')
            else:
                pd.concat(fixed_power).groupby('t').sum().to_csv('SLP.csv', sep=';', decimal=',')

    except Exception as e:
        print(repr(e))
        logger.error(f' -> error while slp or power flow calculation: {repr(e)}')

    # -> start simulation for date range start_date till end_date
    logger.info(' -> starting simulation')
    for day in pd.date_range(start=start_date, end=end_date, freq='d'):
        logger.info(f' -> running day {day.date()}')
        for d_time in tqdm(pd.date_range(start=day, periods=1440, freq='min')):
            try:
                for id_, request in FlexProvider.get_requests(d_time).items():
                    price, utilization, sub_id = CapProvider.get_price(request, d_time)
                    if FlexProvider.commit(id_, request, price, sub_id):
                        for node_id, parameters in request.items():
                            for power, duration in parameters:
                                CapProvider.fixed_power[node_id][d_time:d_time + td(minutes=duration)] += power
                        CapProvider.set_utilization()
                FlexProvider.simulate(d_time)

            except Exception as e:
                print(repr(e))
                logger.error(f' -> error during simulation: {repr(e)}')

    # -> collect results
    try:
        sim_data, car_data = FlexProvider.get_results()
        sim_data.to_sql('vars', engine, index=False, if_exists='append')
        if car_data:
            car_data.to_sql('cars', engine,  index=False, if_exists='append')
    except Exception as e:
        logger.error(repr(e))
        raise Exception(f"Can't store data from FlexProvider in database {DATABASE_URI}")

    try:
        lines, transformers, = CapProvider.get_results()
        for line in lines:
            pd.DataFrame(line).to_sql('grid', engine, index=False, if_exists='append')
        for transformer in transformers:
            pd.DataFrame(transformer).to_sql('grid', engine, index=False, if_exists='append')
    except Exception as e:
        logger.error(repr(e))
        raise Exception(f"Can't store data from CapProvider in database {DATABASE_URI}")
