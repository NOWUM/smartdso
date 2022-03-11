import uuid
import pandas as pd
import numpy as np
from datetime import timedelta, datetime
import logging
import sqlite3


from participants.residential import HouseholdModel
from participants.business import BusinessModel
from agents.utils import WeatherGenerator
from agents.analyser import Check

allocated_consumers = pd.read_csv(r'./gridLib/data/grid_allocations.csv', index_col=0)
nodes = pd.read_csv(r'./gridLib/data/export/nodes.csv', index_col=0)


class FlexibilityProvider:

    def __init__(self, nuts_code: str = 'DEA26'):
        self.participants = {}
        # ---> create participants
        for index, consumer in allocated_consumers.iterrows():
            if consumer['bus0'] in nodes.index and consumer['profile'] == 'H0':
                participant = HouseholdModel(T=96, demandP=consumer['jeb'], grid_node=consumer['bus0'],
                                             residents=int(max(round(consumer['jeb'] / 1500, 0), 1)))
                self.participants[uuid.uuid1()] = participant
            else:
                participant = BusinessModel(T=96, demandP=consumer['jeb'], grid_node=consumer['bus0'])
                self.participants[uuid.uuid1()] = participant
        # ---> set logger
        self._logger = logging.getLogger('FlexibilityProvider')
        self._logger.setLevel('INFO')
        self._logger.info(f'created {len(allocated_consumers)} participants')
        # ---> set weather parameters
        self._nuts3 = nuts_code
        self._weather_generator = WeatherGenerator()

    def _generate_weather(self, d_time: datetime):
        weather = self._weather_generator.get_weather(d_time.replace(year=1996), self._nuts3)
        self._logger.info(f'get weather forecast for the next day {d_time.date()} and forwarding it to the consumers')
        for participant in self.participants.values():
            participant.set_parameter(weather=weather.copy(), prices={})

    def get_fixed_power(self, d_time: datetime):
        self._logger.info('collect demand from participants')
        self._generate_weather(d_time)
        total_powers = []
        for id_, participant in self.participants.items():
            # ---> build dataframe and set index
            df = pd.DataFrame({'power': participant.get_fixed_power(d_time)})
            df.index = pd.date_range(start=d_time, freq='15min', periods=len(df))
            df = df.iloc[:-1]
            # ---> set consumer and grid id
            df['id_'] = str(id_)
            df['node_id'] = participant.grid_node
            df = df.rename_axis('t')
            # ---> add to total power
            total_powers.append(df)
        # ---> build total dataframe and write to db
        df = pd.concat(total_powers)
        return df

    def get_requests(self, d_time: datetime):
        # ---> collection of all requests
        requests = {}
        for id_, participant in self.participants.items():
            request = participant.get_request(d_time)
            if len(request.values()) > 0:
                # print(request)
                requests[id_] = request
        return requests

    def set_demand(self, request: pd.DataFrame = None, id_=None):
        request = request.reset_index()
        request['id_'] = str(id_)

        request = request.set_index(['id_', 'node_id', 't'])

        t1 = request.index.get_level_values('t').min()
        t2 = request.index.get_level_values('t').max()

        df = pd.read_sql(f'Select * from daily_demand where id_=\'{str(id_)}\' '
                         f'and t >= {t1} and t <= {t2}', self._database)

        df = df.set_index(['id_', 'node_id', 't'])
        df.loc[request.index] += request

        for index, data in df.iterrows():
            query = f'Update daily_demand Set power={data.values[0]} ' \
                    f'where id_=\'{index[0]}\' and node_id=\'{index[1]}\' and t={index[2]}'
            self._database.execute(query)
        self._database.commit()


if __name__ == "__main__":
    fp = FlexibilityProvider()
    x = fp.get_fixed_power(pd.to_datetime('2022-02-01'))
    #for t in tqdm(pd.date_range(start='2022-02-01', freq='min', periods=1440)):
    #    r = fp.get_requests(t)
    #    print(r)
    #    for participant in fp.participants.values():
    #        participant.do(t)

