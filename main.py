from agents.flexibility_provider import FlexibilityProvider
from agents.capacity_provider import CapacityProvider
import sqlite3
import pandas as pd
import numpy as np
from tqdm import tqdm
import logging
import os

start_date = os.getenv('START_DATE', '2022-02-01')
end_date = os.getenv('END_DATE', '2022-02-10')

logger = logging.getLogger('Simulation')
logger.setLevel('INFO')

FlexProvider = FlexibilityProvider()
print(FlexProvider)
logging.getLogger('FlexibilityProvider').setLevel('WARNING')

CapProvider = CapacityProvider()
logging.getLogger('CapacityProvider').setLevel('WARNING')

participants = FlexProvider.participants
ref_car = None
while ref_car is None:
    key = np.random.choice([key for key in participants.keys()])
    participant = participants[key]
    for resident in participant.residents:
        if resident.own_car and resident.car.type == 'ev':
            ref_car = resident.car

total_capacity = 0
for participant in participants.values():
    for resident in participant.residents:
        if resident.own_car and resident.car.type == 'ev':
            total_capacity += resident.car.capacity

database = os.getenv('DATABASE', 'result.db')
database = sqlite3.connect(fr'./sim_result/{database}')
database.execute('DROP TABLE IF EXISTS results')


def get_data():
    sim = sqlite3.connect('simulation.db')
    query = 'Select t, sum(power) as power from daily_demand group by node_id, t, id_'
    p = pd.read_sql(query, sim)
    p = p.groupby('t').sum()
    p = pd.Series(p.values.reshape(-1), index=range(1440))
    return p


if __name__ == "__main__":

    # ---> start simulation for date range start_date till end_date
    for day in pd.date_range(start=start_date, end=end_date, freq='d'):
        logger.info(f'start simulation for day {day.date()}')
        # ---> initialize time series to analyse the results
        daily_result = {key: pd.Series(data=np.zeros(1440), index=range(1440))
                        for key in ['commits', 'requests', 'charged', 'soc', 'price', 'ref_distance', 'ref_soc']}

        # ---> set mobility demand for the current day
        for participant in participants.values():
            participant.set_mobility(day)
        # ---> get fixed demand of each household
        daily_demand = FlexProvider.get_fixed_demand(day)
        daily_demand = daily_demand.groupby(['node_id', 't']).sum()
        power = daily_demand.groupby('t').sum()
        power = pd.Series(power.values.reshape(-1), index=range(1440))
        # ---> forward the data to the capacity provider
        CapProvider.plan_fixed_demand(daily_demand=daily_demand.reset_index())
        counter = 0
        for d_time in tqdm(pd.date_range(start=day, periods=1440, freq='min')):
            # ---> for each time step get the charging requests
            requests = FlexProvider.get_requests(d_time)
            daily_result['requests'][counter] = len(requests)
            committed = 0
            # ---> for each request get the price from the capacity provider
            for id_, request in requests.items():
                # ---> get price
                price = CapProvider.get_price(request)
                # ---> forward the price to the consumer
                if participants[id_].commit_charging(price):
                    # ---> lock demand if the charging process is committed
                    FlexProvider.set_demand(request, id_)
                    daily_result['price'][counter] = price/(request['power'].sum()/60)
                    committed += 1
            daily_result['commits'][counter] = committed
            capacity = 0
            # ---> do mobility
            for participant in participants.values():
                daily_result['charged'][counter] += participant.demand['charged'][counter]
                participant.move(d_time)
                # ---> get current soc for each ev
                for resident in participant.residents:
                    if resident.own_car and resident.car.type == 'ev':
                        capacity += resident.car.capacity * resident.car.soc/100
            daily_result['ref_soc'][counter] = ref_car.soc
            daily_result['ref_distance'][counter] = ref_car.total_distance

            # ---> calculate summarized soc
            daily_result['soc'][counter] = capacity/total_capacity
            # ---> increment counter
            counter += 1

        charged = daily_result['charged'] + (get_data() - power)
        power = power - daily_result['charged']
        # ---> save results
        result = pd.DataFrame(dict(power=power.values.flatten(),
                                   charged=charged.values.flatten(),
                                   requests=daily_result['requests'].values,
                                   commits=daily_result['commits'].values,
                                   soc=daily_result['soc'].values,
                                   price=daily_result['price'].values,
                                   ref_soc=daily_result['ref_soc'].values,
                                   ref_distance=daily_result['ref_distance'].values))

        result.index = pd.date_range(start=day, periods=1440, freq='min')
        result.to_sql('results', database, if_exists='append')

database.close()