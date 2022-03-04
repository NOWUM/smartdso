import uuid
import pandas as pd
import numpy as np
from tqdm import tqdm
from datetime import timedelta, datetime
import logging
import sqlite3
from collections import defaultdict

from participants.residential import HouseholdModel
from participants.business import BusinessModel
from agents.utils import WeatherGenerator

allocated_consumers = pd.read_csv(r'./gridLib/data/grid_allocations.csv', index_col=0)
nodes = pd.read_csv(r'./gridLib/data/export/nodes.csv', index_col=0)


class FlexibilityProvider:

    def __init__(self, nuts_code: str = 'DEA26', tqdm: bool = False):
        self._logger = logging.getLogger('FlexibilityProvider')
        self._logger.setLevel('INFO')
        self.participants = {}

        self.summary = dict(
            population=0,
            adults=0,
            employees=0,
            others=0,
            total_vehicles=0,
            electric_vehicles=0,
            mobility_behavior=dict(
                days=dict(
                    work=defaultdict(int),
                    errand=defaultdict(int),
                    hobby=defaultdict(int)
                ),
                times=dict(
                    work=defaultdict(int),
                    errand=defaultdict(int),
                    hobby=defaultdict(int)
                ),
                distances=dict(
                    work=0,
                    errand=0,
                    hobby=0
                ),
                travel=dict(
                    work=0,
                    errand=0,
                    hobby=0
                )
            )
        )

        for index, consumer in allocated_consumers.iterrows():
            if consumer['bus0'] in nodes.index and consumer['profile'] == 'H0':
                participant = HouseholdModel(T=1440, demandP=consumer['jeb'], grid_node=consumer['bus0'],
                                             residents=int(max(round(consumer['jeb'] / 1500, 0), 1)))
                self.summary['population'] += len(participant.residents)
                for resident in participant.residents:
                    self._check_attributes(resident)

                self.participants[uuid.uuid1()] = participant
            else:
                participant = BusinessModel(T=1440, demandP=consumer['jeb'], grid_node=consumer['bus0'])
                self.participants[uuid.uuid1()] = participant

        self._logger.info(f'create {len(allocated_consumers)} participants')
        self._nuts3 = nuts_code
        self._weather_generator = WeatherGenerator()
        self._database = sqlite3.connect('simulation.db')
        self._database.execute(f"DROP TABLE IF EXISTS daily_demand")
        self._tqdm = not tqdm

        for key in ['work', 'errand', 'hobby']:
            data = self.summary['mobility_behavior']['days'][key]
            df = pd.DataFrame.from_dict(data, orient='index').sort_index()
            df['dow'] = [pd.to_datetime(i).dayofweek for i in df.index]
            df = df.sort_values('dow')
            df.index = df.index.day_name()
            del df['dow']
            total = df.sum().values[0]
            self.summary['mobility_behavior']['days'][key] = df/df.sum() * 100

            data = self.summary['mobility_behavior']['times'][key]
            df = pd.DataFrame.from_dict(data, orient='index').sort_index()
            df.index = df.index

            times = {
                '05:00-08:00': df.loc[np.logical_and(5 <= df.index.hour, df.index.hour < 8)].sum().values[0],
                '08:00-10:00': df.loc[np.logical_and(8 <= df.index.hour, df.index.hour < 10)].sum().values[0],
                '10:00-13:00': df.loc[np.logical_and(10 <= df.index.hour, df.index.hour < 13)].sum().values[0],
                '13:00-16:00': df.loc[np.logical_and(13 <= df.index.hour, df.index.hour < 16)].sum().values[0],
                '16:00-19:00': df.loc[np.logical_and(16 <= df.index.hour, df.index.hour < 19)].sum().values[0],
                '19:00-22:00': df.loc[np.logical_and(19 <= df.index.hour, df.index.hour < 22)].sum().values[0],
                '22:00-05:00': df.loc[np.logical_or(df.index.hour < 5, df.index.hour >= 22)].sum().values[0],
            }

            df = pd.DataFrame.from_dict(times, orient='index')

            self.summary['mobility_behavior']['travel'][key] = self.summary['mobility_behavior']['distances'][key] \
                                                               / self.summary['mobility_behavior']['travel'][key] * 60

            self.summary['mobility_behavior']['distances'][key] /= total
            self.summary['mobility_behavior']['times'][key] = df/df.sum() * 100

    def _check_attributes(self, resident):
        # ----> check person attributes
        if resident.type == 'adult':
            self.summary['adults'] += 1
            if resident.employee:
                self.summary['employees'] += 1
        else:
            self.summary['others'] += 1
        # ----> check car attributes
        if resident.own_car:
            self.summary['total_vehicles'] += 1
            if resident.car.type == 'ev':
                self.summary['electric_vehicles'] += 1
        # ----> check mobility pattern
        for date in pd.date_range(start='2018-02-01', periods=7, freq='d'):
            mobility_events = resident.get_mobility_demand(date)
            for e in mobility_events:
                self.summary['mobility_behavior']['days'][e['type']][date] += 1
                t = datetime.strptime(e['start_time'], '%H:%M:%S')
                self.summary['mobility_behavior']['times'][e['type']][t] += 1
                self.summary['mobility_behavior']['distances'][e['type']] += e['distance']
                self.summary['mobility_behavior']['travel'][e['type']] += e['travel_time']

    def _generate_weather(self, d_time: datetime):
        weather = self._weather_generator.get_weather(d_time.replace(year=1996), self._nuts3)
        self._logger.info(f'get weather forecast for the next day {d_time.date()} and forwarding it to the consumers')
        for participant in tqdm(self.participants.values(), disable=self._tqdm):
            participant.set_parameter(weather=weather.copy(), prices={})

    def get_fixed_demand(self, d_time: datetime):
        self._logger.info('reset table daily_demand')
        self._generate_weather(d_time)
        demand = {}
        for id_, participant in tqdm(self.participants.items(), disable=self._tqdm):
            participant.get_fixed_demand(d_time)
            for t in range(1440):
                demand[(str(id_), participant.grid_node, t)] = participant.demand['power'][t]

        df = pd.DataFrame.from_dict({'power': demand}, orient='columns')
        df = df.rename_axis(['id_', 'node_id', 't'])

        df.to_sql('daily_demand', self._database, if_exists='replace')

        return df

    def get_requests(self, d_time: datetime):
        # ---> collection of all requests
        requests = {}
        for id_, participant in tqdm(self.participants.items(), disable=self._tqdm):
            request = participant.get_request(d_time)
            # ---> add if only a request charging power is required
            if not request.empty:
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

    def __str__(self):
        string = f'############################## FlexProvider ############################## \n' \
                 f'total population: {self.summary["population"]} \n' \
                 f'residents with driver licence: {self.summary["adults"]} \n' \
                 f'residents without driver licence: {self.summary["others"]}  \n' \
                 f'employees: {self.summary["employees"]}  \n' \
                 f'total cars: {self.summary["total_vehicles"]}  \n' \
                 f'electric vehicles: {self.summary["electric_vehicles"]} \n \n' \

        string += f'Mobility Pattern \n' \

        for key in ['work', 'errand', 'hobby']:
            df = self.summary['mobility_behavior']['days'][key]

            string += f'-> Type: {key} \n'

            string += 'Days: \n'
            for index, row in df.iterrows():
                value = row[0]
                for bar in range(0, int(value)):
                    string += f'||'
                string += f'  {index} {round(value, 2)} % \n'

            string += 'Times: \n'
            df = self.summary['mobility_behavior']['times'][key]
            for index, row in df.iterrows():
                value = row[0]
                for bar in range(0, int(value)):
                    string += f'||'
                string += f'  {index} {round(value, 2)} % \n'

            string += f'mean distance: {round(self.summary["mobility_behavior"]["distances"][key],2)} km \n'
            string += f'mean travel time: {round(self.summary["mobility_behavior"]["travel"][key], 2)} km/h \n'
            string += '\n'

        return string

if __name__ == "__main__":
    fp = FlexibilityProvider()
    d = fp.get_fixed_demand(pd.to_datetime('2018-01-01'))
    r = fp.get_requests(pd.to_datetime('2018-01-01'))
