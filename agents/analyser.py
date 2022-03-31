from collections import defaultdict
import pandas as pd
import numpy as np
from datetime import datetime


class Check:

    def __init__(self, type_: str = 'FlexibilityProvider'):
        self.type_ = type_
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
        for day, events in resident.mobility_generator.mobility.items():
            for event in events:
                self.summary['mobility_behavior']['days'][event['type']][day] += 1
                t = datetime.strptime(event['start_time'], '%H:%M:%S')
                self.summary['mobility_behavior']['times'][event['type']][t] += 1
                self.summary['mobility_behavior']['distances'][event['type']] += event['distance']
                self.summary['mobility_behavior']['travel'][event['type']] += event['travel_time']

    def run(self, agent):
        if self.type_ == 'FlexibilityProvider':
            for participant in agent.clients.values():
                for resident in participant.persons:
                    self._check_attributes(resident)

            for key in ['work', 'errand', 'hobby']:
                data = self.summary['mobility_behavior']['days'][key]
                df = pd.DataFrame.from_dict(data, orient='index')
                total = df.sum().values[0]
                self.summary['mobility_behavior']['days'][key] = df / df.sum() * 100

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
                                                                   / self.summary['mobility_behavior']['travel'][
                                                                       key] * 60

                self.summary['mobility_behavior']['distances'][key] /= total
                self.summary['mobility_behavior']['times'][key] = df / df.sum() * 100

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
