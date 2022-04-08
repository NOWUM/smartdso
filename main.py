import pandas as pd
import numpy as np
from tqdm import tqdm
import logging
import os
from datetime import timedelta as td
from pathlib import Path

from agents.flexibility_provider import FlexibilityProvider
from agents.capacity_provider import CapacityProvider

logger = logging.getLogger('Simulation')
logger.setLevel('INFO')

start_date = pd.to_datetime(os.getenv('START_DATE', '2022-01-01'))
end_date = pd.to_datetime(os.getenv('END_DATE', '2022-01-5'))
logger.info(f' --> simulation for horizon {start_date.date} till {end_date.date}')
scenario_name = os.getenv('SCENARIO_NAME', 'base')
logger.info(f' --> scenario {scenario_name}')

path = os.getenv('RESULT_PATH', 'base')

input_set = {'london_data': (os.getenv('LONDON_DATA', 'True') == 'True'),
             'minimum_soc': int(os.getenv('MINIMUM_SOC', -1)),
             'start_date': start_date,
             'end_date': end_date,
             'ev_ratio': int(os.getenv('EV_RATIO', 100))/100,
             'base_price': int(os.getenv('BASE_PRICE', 29))}

logger.info(' --> starting Flexibility Provider')
FlexProvider = FlexibilityProvider(**input_set)
logging.getLogger('FlexibilityProvider').setLevel('WARNING')
logger.info(' --> starting Capacity Provider')
CapProvider = CapacityProvider(**input_set)
logging.getLogger('CapacityProvider').setLevel('WARNING')

logger.info(' --> initialize result set')
len_ = 1440 * ((end_date - start_date).days + 1)
time_range = pd.date_range(start=start_date, periods=len_, freq='min')
result = {key: pd.Series(data=np.zeros(len_), index=range(len_)) for key in ['charged', 'shift', 'price']}

if __name__ == "__main__":
    try:
        # --> run SLPs for each day in simulation horizon
        logger.info(f' --> running slp - generation for {start_date.date()} till {end_date.date()}')
        fixed_power = []
        for day in tqdm(pd.date_range(start=start_date, end=end_date, freq='d')):
            fixed_power += [FlexProvider.get_fixed_power(day)]
        # --> forward the slp data to the Capacity Provider
        logger.info(f' --> running power flow calculation for {start_date.date()} till {end_date.date()}')
        CapProvider.set_fixed_power(data=pd.concat(fixed_power))
    except Exception as e:
        print(repr(e))
        logger.error(f' --> error in slp generation: {repr(e)}')

    # --> start simulation for date range start_date till end_date
    logger.info(' --> starting mobility simulation')
    indexer = 0                                                                 # --> minute counter for result set
    for day in pd.date_range(start=start_date, end=end_date, freq='d'):
        logger.info(f' --> simulation for day {day.date()}')
        # --> build dictionary to save simulation results
        for d_time in tqdm(pd.date_range(start=day, periods=1440, freq='min')):
            try:
                requests = FlexProvider.get_requests(d_time)
                for id_, request in requests.items():
                    price, utilization, sub_id = CapProvider.get_price(request, d_time)
                    commit, w_time = FlexProvider.commit(id_, price, d_time)
                    if commit:
                        result['price'][indexer] = price + input_set['base_price']
                        for node_id, parameters in request.items():
                            for power, duration in parameters:
                                result['charged'][indexer:indexer + duration] += power
                                CapProvider.fixed_power[node_id][d_time:d_time + td(minutes=duration)] += power
                                if w_time > 0:
                                    result['shift'][indexer:indexer + duration] += power
                        CapProvider.set_charging(price, utilization, sub_id, d_time)

                FlexProvider.simulate(d_time)

            except Exception as e:
                print(repr(e))
                logger.error(f' --> error in simulation: {repr(e)}')
            indexer += 1

# --> collect results
result_set = pd.DataFrame(result)
result_set['price'] = result_set['price'].replace(to_replace=0, method='ffill')
for key in CapProvider.utilization.keys():
    result_set[f'utilization {key}'] = CapProvider.utilization[key].replace(to_replace=0, method='ffill').values
    result_set[f'price {key}'] = CapProvider.price[key].values + input_set['base_price']
    result_set[f'congestion {key}'] = CapProvider.congestion[key].values
result_set['soc'] = FlexProvider.soc
result_set['ref_soc'] = FlexProvider.ref_soc
result_set['ref_distance'] = FlexProvider.ref_distance
result_set['empty'] = FlexProvider.empty_counter
result_set['concurrency'] = result_set['charged']/FlexProvider.power
result_set.index = time_range
# --> save results
path_name = fr'./sim_result/S_{path}'
logger.info(f' --> saving results in {path_name}')
if not Path(fr'{path_name}').is_dir():
    os.mkdir(fr'{path_name}')
sim = scenario_name.split('_')[-1]
result_set.to_csv(fr'{path_name}/result_1min_{sim}.csv', sep=';', decimal=',')
