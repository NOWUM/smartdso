import pandas as pd
from tqdm import tqdm
import logging
import os
from datetime import timedelta as td
from sqlalchemy import create_engine, inspect

from agents.flexibility_provider import FlexibilityProvider
from agents.capacity_provider import CapacityProvider

logger = logging.getLogger('Simulation')
logger.setLevel('INFO')

start_date = pd.to_datetime(os.getenv('START_DATE', '2022-01-01'))
end_date = pd.to_datetime(os.getenv('END_DATE', '2022-01-15'))
logger.info(f' -> simulation for horizon {start_date.date} till {end_date.date}')
scenario_name = os.getenv('SCENARIO_NAME', 'EV100LIMIT-1L_0')
sim = os.getenv('RESULT_PATH', scenario_name.split('_')[-1])
logger.info(f' -> scenario {scenario_name}')

input_set = {'london_data': (os.getenv('LONDON_DATA', 'False') == 'True'),
             'dynamic_fee': (os.getenv('DYNAMIC_FEE', 'True') == 'True'),
             'minimum_soc': int(os.getenv('MINIMUM_SOC', -1)),
             'start_date': start_date,
             'end_date': end_date,
             'ev_ratio': int(os.getenv('EV_RATIO', 100))/100,
             'base_price': int(os.getenv('BASE_PRICE', 29))}

logger.info(' -> starting Flexibility Provider')
FlexProvider = FlexibilityProvider(**input_set)
logging.getLogger('FlexibilityProvider').setLevel('WARNING')
logger.info(' -> starting Capacity Provider')
CapProvider = CapacityProvider(**input_set)
logging.getLogger('CapacityProvider').setLevel('WARNING')

logger.info(' -> connecting to database')
engine = create_engine('postgresql://opendata:opendata@10.13.10.41:5432/smartdso')
tables = inspect(engine).get_table_names()

if __name__ == "__main__":
    try:
        # -> run SLPs for each day in simulation horizon
        logger.info(f' -> running slp - generation for {start_date.date()} till {end_date.date()}')
        fixed_power = []
        for day in tqdm(pd.date_range(start=start_date, end=end_date, freq='d')):
            fixed_power += [FlexProvider.get_fixed_power(day)]
        # -> forward the slp data to the Capacity Provider
        logger.info(f' -> running power flow calculation for {len(CapProvider.mapper.unique())} grids')
        CapProvider.set_fixed_power(data=pd.concat(fixed_power))
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


# --> collect results
car_data, s_car_data, sim_data = FlexProvider.get_results()
table = dict(cars=car_data, meta=sim_data, evs=s_car_data)
for key, value in table.items():
    value['iteration'] = sim
    value['scenario'] = scenario_name.split('_')[0]
    value = value.set_index(['time', 'iteration', 'scenario'])
    value.to_sql(key, engine, if_exists='append', index=True, index_label=['time', 'iteration', 'scenario'])

lines, line_mapping, transformers, transformer_mapping = CapProvider.get_results()
table = dict(lines=lines, transformers=transformers)
mapping = dict(lines=line_mapping, transformers=transformer_mapping)
for key, value in table.items():
    value = pd.Series({(time, sim, scenario_name.split('_')[0], asset, mapping[key][asset]): utilization
                       for time, values in value.iterrows()
                       for asset, utilization in values.items()})
    value = value.reset_index()
    value.columns = ['time', 'iteration', 'scenario', 'id_', 'grid', 'utilization']
    value = value.set_index(['time', 'iteration', 'scenario', 'id_', 'grid'])
    value.to_sql(key, engine, if_exists='append', index=True, index_label=['time', 'iteration',
                                                                           'scenario', 'id_', 'grid'])

