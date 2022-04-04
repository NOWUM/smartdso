from mobLib.utils import read_MIT_data
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

pd.options.mode.chained_assignment = None

MIT_path = r'./mobLib/data/'
work, errand, hobby = read_MIT_data(MIT_path)


def normalize_probabilities(probabilities: list):
    probabilities = np.asarray(probabilities)
    probabilities /= probabilities.sum()
    return probabilities


class MobilityDemand:

    def __init__(self, demand_types: list = None):
        self.demand_types = demand_types  # ---> work, errand or hobby
        self.working_days = []  # ---> list of working days
        self.errand_days = []  # ---> list of errand days
        self.hobby_days = []  # ---> list of leisure days
        self.car_usage = False  # ---> car required for mobility
        # ---> mapping for days to get the right order
        self._day_map = dict(Monday=0, Tuesday=1, Wednesday=2, Thursday=3, Friday=4, Saturday=5, Sunday=6)
        self.employee_relation = None
        self.mobility = {key: [] for key in self._day_map.keys()}
        for demand_type in self.demand_types:
            if demand_type == 'work':
                self._get_mobility_demand(tables=work, mobility_type=demand_type)
            elif demand_type == 'hobby':
                self._get_mobility_demand(tables=hobby, mobility_type=demand_type)
            elif demand_type == 'errand':
                self._get_mobility_demand(tables=errand, mobility_type=demand_type)

        max_distance = 0
        for demands in self.mobility.values():
            distance = 0
            for demand in demands:
                distance += demand['distance'] * 2
            if distance > max_distance:
                max_distance = distance

        self.maximal_distance = max_distance

    def _get_days(self, days: list, probabilities: list, size: int):
        probabilities = normalize_probabilities(probabilities)
        return sorted(np.random.choice(a=days, p=probabilities, size=size, replace=False),
                      key=self._day_map.get)

    def _get_start_time(self, time_intervals: list, probabilities: list):
        probabilities = normalize_probabilities(probabilities)
        time_interval = np.random.choice(a=time_intervals, p=probabilities, replace=False)
        t1 = datetime.strptime(time_interval.split('-')[0], '%H:%M')
        t2 = datetime.strptime(time_interval.split('-')[1], '%H:%M')
        time_interval = list(pd.date_range(start=t1, end=t2 if t2 > t1 else t2 + timedelta(days=1), freq='15min'))

        start_time = np.random.choice(time_interval)

        return start_time.time().isoformat()

    def _get_distance(self, distance_intervals: list, probabilities: list):
        probabilities = normalize_probabilities(probabilities)
        distance_interval = np.random.choice(a=distance_intervals, p=probabilities, replace=False)

        if '>' in distance_interval:
            d = distance_interval.split('>')[-1]
            d = float(d.replace(' ', ''))
            distance = np.random.poisson(lam=0.01 * d) + d
        elif '<' in distance_interval:
            d = distance_interval.split('<')[-1]
            d = float(d.replace(' ', ''))
            distance = np.random.poisson(lam=0.5 * d) + 0.1
        else:
            distance = distance_interval.split('-')
            distance = np.random.uniform(low=float(distance[0]), high=float(distance[-1]))

        return round(distance, 1), distance_interval

    def _get_travel_time(self, time_intervals: list, probabilities: list):
        probabilities = normalize_probabilities(probabilities)
        time_interval = np.random.choice(a=time_intervals, p=probabilities, replace=False)

        if '>' in time_interval:
            time_interval = time_interval.split('>')[-1]
            time_interval = int(time_interval.replace(' ', ''))
            duration = np.random.poisson(lam=0.1 * time_interval) + time_interval
        elif '<' in time_interval:
            time_interval = time_interval.split('<')[-1]
            time_interval = int(time_interval.replace(' ', ''))
            duration = np.random.poisson(lam=0.5 * time_interval) + 2.5
        else:
            time_interval = time_interval.split('-')
            duration = np.random.uniform(low=float(time_interval[0]), high=float(time_interval[-1]))

        return round(duration, 0)

    def _get_job_type(self, job_types: list, probabilities: list):
        probabilities = normalize_probabilities(probabilities)
        job_type = np.random.choice(a=job_types, p=probabilities, replace=False)
        return job_type

    def _get_means_of_transport(self, means_of_transports: list, probabilities: list):
        probabilities = normalize_probabilities(probabilities)
        travel_by = np.random.choice(a=means_of_transports, p=probabilities, replace=False)
        return travel_by

    def _get_mobility_event(self, tables: dict, mobility_type: str = 'work', in_week: bool = True):

        mobility_event = dict(type=mobility_type)

        if mobility_type == 'work' and self.employee_relation is None:
            job_type = self._get_job_type(tables['type'].index, tables['type'].loc[:, 'Probabilities'])
            duration = 525 if job_type == 'Full_Time' else 240
            self.employee_relation = duration
        elif mobility_type == 'work':
            duration = self.employee_relation
        elif mobility_type == 'errand':
            duration = 35
        else:
            duration = 90

        mobility_event['duration'] = duration
        if in_week:
            mobility_event['start_time'] = self._get_start_time(tables['times'].index, tables['times'].loc[:, 'Week'])
        else:
            mobility_event['start_time'] = self._get_start_time(tables['times'].index,
                                                                tables['times'].loc[:, 'Weekend'])

        distance, index = self._get_distance(tables['distances'].index, tables['distances'].loc[:, 'Probabilities'])
        mobility_event['distance'] = distance
        travel_time = self._get_travel_time(tables['durations'].columns, tables['durations'].loc[index, :])
        mobility_event['travel_time'] = travel_time
        travel_by = self._get_means_of_transport(tables['usage'].columns, tables['usage'].loc[index, :])
        mobility_event['car_use'] = True if travel_by == 'Car' else False
        if mobility_event['car_use']:
            self.car_usage = True

        return mobility_event

    def _check_overlap(self, day: str, mobility_event: dict):
        for event in self.mobility[day]:
            start_time = datetime.strptime(event['start_time'], '%H:%M:%S')
            t_departure = start_time - timedelta(minutes=event['travel_time'])
            t_arrival = start_time + timedelta(minutes=(event['travel_time'] + event['duration']))
            time_range = pd.date_range(start=t_departure, end=t_arrival, freq='min')
            if datetime.strptime(mobility_event['start_time'], '%H:%M:%S') in time_range:
                return True
            end_time = datetime.strptime(mobility_event['start_time'], '%H:%M:%S') \
                       + timedelta(minutes=mobility_event['duration'] + mobility_event['travel_time'])
            if end_time in time_range:
                return True
        return False

    def _get_mobility_demand(self, tables: dict, mobility_type: str = 'work'):
        if mobility_type == 'work':
            days = self._get_days(tables['days'].index, tables['days'].loc[:, 'Probabilities'], 5)
            mobility_event = self._get_mobility_event(tables, mobility_type)
            while len(days) > 0:
                day = days[0]
                if self._day_map[day] > 5:
                    mobility_event = self._get_mobility_event(tables, mobility_type, in_week=False)
                if not self._check_overlap(day, mobility_event):
                    self.mobility[day].append(mobility_event)
                    self.working_days.append(day)
                    days.pop(0)
        else:
            if mobility_type == 'errand':
                number = np.random.choice(a=tables['freq'].index, p=tables['freq'].loc[:, 'Probabilities'])
            else:
                number = 2
            days = self._get_days(tables['days'].index, tables['days'].loc[:, 'Probabilities'], number)
            while len(days) > 0:
                day = days[0]
                if self._day_map[day] > 5:
                    mobility_event = self._get_mobility_event(tables, mobility_type, in_week=False)
                else:
                    mobility_event = self._get_mobility_event(tables, mobility_type)
                if not self._check_overlap(day, mobility_event):
                    self.mobility[day].append(mobility_event)
                    days.pop(0)
                    if mobility_type == 'errand':
                        self.errand_days.append(day)
                    else:
                        self.hobby_days.append(day)


if __name__ == "__main__":
    # for _ in range(1000):
    a = MobilityDemand(demand_types=['work', 'hobby', 'errand'])
