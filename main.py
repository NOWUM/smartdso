from agents.flexibility_provider import FlexibilityProvider
from agents.capacity_provider import CapacityProvider
import sqlite3
import pandas as pd
import numpy as np
from tqdm import tqdm
import logging
import os
from datetime import timedelta as td

start_date = pd.to_datetime(os.getenv('START_DATE', '2022-02-01'))
end_date = pd.to_datetime(os.getenv('END_DATE', '2022-02-02'))

logger = logging.getLogger('Simulation')
logger.setLevel('INFO')

logger.info(' ---> starting Flexibility Provider')
FlexProvider = FlexibilityProvider()
logging.getLogger('FlexibilityProvider').setLevel('WARNING')
logger.info(' ---> starting Capacity Provider')
CapProvider = CapacityProvider()
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

logger.info(' ---> initialize database')
database = os.getenv('DATABASE', 'result.db')
database = sqlite3.connect(fr'./sim_result/{database}')
database.execute('DROP TABLE IF EXISTS results')

len_ = 1440 *((end_date-start_date).days + 1)
time_range = pd.date_range(start=start_date, periods=len_, freq='min')

result = {key: pd.Series(data=np.zeros(len_), index=range(len_))
          for key in ['commits', 'requests', 'charged', 'soc', 'price', 'ref_distance', 'ref_soc']}


def get_data():
    sim = sqlite3.connect('simulation.db')
    query = 'Select t, sum(power) as power from daily_demand group by node_id, t, id_'
    p = pd.read_sql(query, sim)
    p = p.groupby('t').sum()
    p = pd.Series(p.values.reshape(-1), index=range(1440))
    return p


if __name__ == "__main__":

    # ---> run SLPs for each day in simulation horizon
    logger.info(f' ---> running slp - generation for {start_date} till {end_date}')
    fixed_power = []
    for day in tqdm(pd.date_range(start=start_date, end=end_date, freq='d')):
        fixed_power += [FlexProvider.get_fixed_power(day)]
    # ---> forward the slp data to the Capacity Provider
    CapProvider.set_fixed_power(data=pd.concat(fixed_power))

    indexer = 0
    # ---> start simulation for date range start_date till end_date
    logger.info(' ---> starting mobility simulation')
    for day in pd.date_range(start=start_date, end=end_date, freq='d'):
        logger.info(f' #### simulation for day {day.date()} ####')
        # ---> build dictionary to save simulation results
        for d_time in tqdm(pd.date_range(start=day, periods=1440, freq='min')):
            requests = FlexProvider.get_requests(d_time)
            for id_, request in requests.items():
                result['requests'][indexer] += 1
                # ---> get price
                price = CapProvider.get_price(request, d_time)
                if participants[id_].commit_charging(price):
                    # logger.info(f' ---> committed charging - price: {round(price,2)} ct/kWh')
                    result['commits'][indexer] += 1
                    result['price'][indexer] = price

                    for node_id, parameters in request.items():
                        for power, duration in parameters:
                            result['charged'][indexer:indexer + duration] += power
                            CapProvider.fixed_power[node_id][d_time:d_time + td(minutes=duration)] += power
                else:
                    pass
                    # logger.info(f' ---> rejected charging - price: {round(price,2)} ct/kWh')
            capacity = 0
            for participant in participants.values():
                participant.do(d_time)
                if len(participant.residents) > 0:
                    for value in participant.car_manager.values():
                        capacity += value['car'].soc/100 * value['car'].capacity
            result['soc'][indexer] = (capacity/total_capacity) * 100
            result['ref_soc'][indexer] = ref_car.soc
            result['ref_distance'][indexer] = ref_car.total_distance
            indexer += 1
            # logger.info(f'SoC: {ref_car.soc}')


result_set = pd.DataFrame(result)
result_set.to_csv('result.csv', sep=';', decimal=',')