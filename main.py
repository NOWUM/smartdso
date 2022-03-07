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

    return p


if __name__ == "__main__":

    # ---> start simulation for date range start_date till end_date
    for day in pd.date_range(start=start_date, end=end_date, freq='d'):
        logger.info(f'start simulation for day {day.date()}')
        # ---> initialize time series to analyse the results
        commits = pd.Series(data=np.zeros(1440), index=range(1440))
        requests_ = pd.Series(data=np.zeros(1440), index=range(1440))
        soc = pd.Series(data=np.zeros(1440), index=range(1440))
        price_ = pd.Series(data=np.zeros(1440), index=range(1440))
        # ---> set mobility demand for the current day
        for participant in participants.values():
            participant.set_mobility(day)
        # ---> get fixed demand of each household
        daily_demand = FlexProvider.get_fixed_demand(day)
        daily_demand = daily_demand.groupby(['node_id', 't']).sum()
        power = daily_demand.groupby('t').sum()
        # ---> forward the data to the capacity provider
        CapProvider.plan_fixed_demand(daily_demand=daily_demand.reset_index())
        counter = 0
        for d_time in tqdm(pd.date_range(start=day, periods=1440, freq='min')):
            # ---> for each time step get the charging requests
            requests = FlexProvider.get_requests(d_time)
            requests_[counter] = len(requests)
            committed = 0
            # ---> for each request get the price from the capacity provider
            for id_, request in requests.items():
                # ---> get price
                price = CapProvider.get_price(request)
                # ---> forward the price to the consumer
                if participants[id_].commit_charging(price):
                    # ---> lock demand if the charging process is committed
                    FlexProvider.set_demand(request, id_)
                    price_[counter] = price/(request['power'].sum()/60)
                    committed += 1
            commits[counter] = committed
            capacity = 0
            # ---> do mobility
            for participant in participants.values():
                participant.move(d_time)
                # ---> get current soc for each ev
                for resident in participant.residents:
                    if resident.own_car and resident.car.type == 'ev':
                        capacity += resident.car.capacity * resident.car.soc/100
            # ---> calculate summarized soc
            soc[counter] = capacity/total_capacity
            # ---> increment counter
            counter += 1

        charged = get_data() - power
        # ---> save results
        result = pd.DataFrame(dict(power=power.values.flatten(),
                                   charged=charged.values.flatten(),
                                   requests=requests_.values,
                                   commits=commits.values,
                                   soc=soc.values,
                                   price=price_.values))
        result.index = pd.date_range(start=day, periods=1440, freq='min')
        result.to_sql('results', database, if_exists='append')

database.close()