import glob
import pandas as pd
import numpy as np
from tqdm import tqdm
import logging
import os
from datetime import timedelta as td
from collections import defaultdict
from plotly.offline import plot
import plotly.express as px
from pathlib import Path
import shutil


from agents.flexibility_provider import FlexibilityProvider
from agents.capacity_provider import CapacityProvider

logger = logging.getLogger('Simulation')
logger.setLevel('INFO')

start_date = pd.to_datetime(os.getenv('START_DATE', '2022-01-01'))
end_date = pd.to_datetime(os.getenv('END_DATE', '2022-01-02'))
logger.info(f' ---> simulation for horizon {start_date.date} till {end_date.date}')
scenario_name = os.getenv('SCENARIO_NAME', 'base')
logger.info(f' ---> scenario {scenario_name}')

path = os.getenv('RESULT_PATH', 'base')

input_set = {'employee_ratio': os.getenv('EMPLOYEE_RATIO', 0.7),
             'london_data': (os.getenv('LONDON_DATA', 'False') == 'True'),
             'minimum_soc': int(os.getenv('MINIMUM_SOC', 50)),
             'start_date': start_date,
             'end_date': end_date,
             'ev_ratio': int(os.getenv('EV_RATIO', 50))/100}

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
          for key in ['commits', 'rejects', 'requests', 'waiting', 'charged', 'shift',
                      'soc', 'price', 'ref_distance', 'ref_soc']}
waiting_time = defaultdict(list)

lmp = {node: pd.Series(data=np.zeros(len_), index=range(len_)) for node in CapProvider.grid.data['connected'].index}

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
                commit, wait = participants[id_].commit_charging(price, d_time)
                if commit:
                    # logger.info(f' ---> committed charging - price: {round(price,2)} ct/kWh')
                    result['commits'][indexer] += 1
                    result['price'][indexer] = price
                    for node_id, parameters in request.items():
                        for power, duration in parameters:
                            if wait == 0:
                                result['charged'][indexer:indexer + duration] += power
                                waiting_time[indexer].append(wait)
                            else:
                                result['shift'][indexer:indexer + duration] += power
                                waiting_time[indexer-wait].append(wait)
                            CapProvider.fixed_power[node_id][d_time:d_time + td(minutes=duration)] += power
                            lmp[node_id][indexer] = price
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
for index, value in waiting_time.items():
    result['waiting'][index] = np.mean(value)


path_name = fr'./sim_result/S_{path}'
logger.info(f'saving results in {path}')
result_name = fr'./sim_result/R_{path}'

if not Path(fr'{path_name}').is_dir():
    os.mkdir(fr'{path_name}')
if not Path(fr'{result_name}').is_dir():
    os.mkdir(fr'{result_name}')

for f in glob.glob(fr'./sim_result/templates/*.xlsx'):
    shutil.copy(f, result_name)

sim = scenario_name.split('_')[-1]

result_set = pd.DataFrame(result)
result_set['price'] = result_set['price'].replace(to_replace=0, method='ffill')
result_set.index = time_range
result_set.to_csv(fr'{path_name}/result_1min_{sim}.csv', sep=';', decimal=',')

resampled_result = result_set.resample('5min').agg({'commits': 'sum', 'rejects': 'sum',
                                                    'requests': 'sum', 'waiting': 'mean',
                                                    'charged': 'mean', 'shift': 'mean',
                                                    'soc': 'mean', 'price': 'mean',
                                                    'ref_distance': 'mean', 'ref_soc': 'mean'})
resampled_result.to_csv(fr'{path_name}/result_5min_{sim}.csv', sep=';', decimal=',')

# ---> save lmp prices
lmp = pd.DataFrame(lmp)
lmp.index = result_set.index
lmp = lmp.loc[:, (lmp != 0).any(axis=0)]
lmp.to_csv(fr'{path_name}/lmp_1min_{sim}.csv', sep=';', decimal=',')
resampled_lmp = lmp.resample('5min').mean()
resampled_lmp.to_csv(fr'{path_name}/lmp_5min_{sim}.csv', sep=';', decimal=',')


plotting = False
if plotting:
    plot_lmp = lmp.resample('60min').mean()
    plot_data = []
    for index in plot_lmp.index:
        for key, value in plot_lmp.loc[index].to_dict().items():
            plot_data.append([index, key, value])
    plot_data = pd.DataFrame(plot_data, columns=['timestamp', 'node', 'price'])
    plot_data['timestamp'] = [pd.to_datetime(str(value)).strftime('%Y-%m-%d %X') for value in plot_data['timestamp'].values]
    figure = px.scatter(plot_data, x="node", y="price", animation_frame="timestamp", size="price", color="node")
    plot(figure, 'temp-plot.html')
