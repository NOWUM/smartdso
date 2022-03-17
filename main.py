from agents.flexibility_provider import FlexibilityProvider
from agents.capacity_provider import CapacityProvider
import pandas as pd
import numpy as np
from tqdm import tqdm
import logging
import os
from datetime import timedelta as td
from collections import defaultdict

logger = logging.getLogger('Simulation')
logger.setLevel('INFO')

start_date = pd.to_datetime(os.getenv('START_DATE', '2022-01-01'))
end_date = pd.to_datetime(os.getenv('END_DATE', '2022-01-02'))
logger.info(f' ---> simulation for horizon {start_date.date} till {end_date.date}')
scenario_name = os.getenv('SCENARIO_NAME', 'base_scenario')
logger.info(f' ---> scenario {scenario_name}')

input_set = {'employee_ratio': os.getenv('EMPLOYEE_RATIO', 0.7),
             'london_data': (os.getenv('LONDON_DATA', 'False') == 'True'),
             'minimum_soc': int(os.getenv('MINIMUM_SOC', 30)),
             'start_date': start_date,
             'end_date': end_date,
             'ev_ratio': int(os.getenv('EV_RATIO', 80))/100}

logger.info(' ---> starting Flexibility Provider')
FlexProvider = FlexibilityProvider(**input_set)
logging.getLogger('FlexibilityProvider').setLevel('WARNING')
logger.info(' ---> starting Capacity Provider')
CapProvider = CapacityProvider(**input_set)
logging.getLogger('CapacityProvider').setLevel('WARNING')

logger.info(' ---> choosing reference car')
participants = FlexProvider.participants
ref_car = None
while ref_car is None:
    key = np.random.choice([key for key in participants.keys()])
    participant = participants[key]
    for resident in participant.residents:
        if resident.own_car and resident.car.type == 'ev':
            ref_car = resident.car

logger.info(' ---> collecting total capacity')
total_capacity = 0
for participant in participants.values():
    for resident in participant.residents:
        if resident.own_car and resident.car.type == 'ev':
            total_capacity += resident.car.capacity

logger.info(' ---> initialize result set')
len_ = 1440 * ((end_date - start_date).days + 1)
time_range = pd.date_range(start=start_date, periods=len_, freq='min')
result = {key: pd.Series(data=np.zeros(len_), index=range(len_))
          for key in ['commits', 'rejects', 'requests', 'charged', 'shift', 'soc', 'price', 'ref_distance', 'ref_soc']}
waiting_time = defaultdict(list)


if __name__ == "__main__":

    # ---> run SLPs for each day in simulation horizon
    logger.info(f' ---> running slp - generation for {start_date} till {end_date}')
    fixed_power = []
    for day in tqdm(pd.date_range(start=start_date, end=end_date, freq='d')):
        fixed_power += [FlexProvider.get_fixed_power(day)]
    # ---> forward the slp data to the Capacity Provider
    CapProvider.set_fixed_power(data=pd.concat(fixed_power))

    # ---> start simulation for date range start_date till end_date
    logger.info(' ---> starting mobility simulation')
    indexer = 0  # ---> minute counter for result set
    for day in pd.date_range(start=start_date, end=end_date, freq='d'):
        logger.info(f' #### simulation for day {day.date()} ####')
        # ---> build dictionary to save simulation results
        for d_time in tqdm(pd.date_range(start=day, periods=1440, freq='min')):
            requests = FlexProvider.get_requests(d_time)
            for id_, request in requests.items():
                result['requests'][indexer] += 1
                # ---> get price
                price = CapProvider.get_price(request, d_time)
                commit, wait = participants[id_].commit_charging(price)
                if commit:
                    # logger.info(f' ---> committed charging - price: {round(price,2)} ct/kWh')
                    result['commits'][indexer] += 1
                    result['price'][indexer] = price
                    for node_id, parameters in request.items():
                        for power, duration in parameters:
                            if wait == 0:
                                result['charged'][indexer:indexer + duration] += power
                            else:
                                result['shift'][indexer:indexer + duration] += power
                                waiting_time[indexer-wait].append(wait)
                            CapProvider.fixed_power[node_id][d_time:d_time + td(minutes=duration)] += power
                else:
                    result['rejects'][indexer] += 1
                    # logger.info(f' ---> rejected charging - price: {round(price,2)} ct/kWh')
            capacity = 0
            for participant in participants.values():
                participant.do(d_time)
                if len(participant.residents) > 0:
                    for value in participant.car_manager.values():
                        capacity += value['car'].soc / 100 * value['car'].capacity
            result['soc'][indexer] = (capacity / total_capacity) * 100
            result['ref_soc'][indexer] = ref_car.soc
            result['ref_distance'][indexer] = ref_car.total_distance
            indexer += 1
            # logger.info(f'SoC: {ref_car.soc}')

# ---> save results
logger.info(f'saving results in ./sim_result/{scenario_name}.csv')
result_set = pd.DataFrame(result)
result_set['price'] = result_set['price'].replace(to_replace=0, method='ffill')
result_set.index = time_range
result_set.to_csv(fr'./sim_result/{scenario_name}.csv', sep=';', decimal=',')
resampled_result = result_set.resample('5min').agg({'commits': 'sum',
                                                    'rejects': 'sum',
                                                    'requests': 'sum',
                                                    'charged': 'mean',
                                                    'shift': 'mean',
                                                    'soc': 'mean',
                                                    'price': 'mean',
                                                    'ref_distance': 'mean',
                                                    'ref_soc': 'mean'})
resampled_result.to_csv(fr'./sim_result/{scenario_name}_resampled.csv', sep=';', decimal=',')
