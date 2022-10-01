from datetime import datetime, timedelta as td
import pandas as pd
import numpy as np
import os

from mobLib.mobility_demand import MobilityDemand

# -> load electric vehicle data
electric_vehicles = pd.read_csv(r'./carLib/data/evs.csv', sep=';', decimal=',')
electric_vehicles['maximal_charging_power'] = electric_vehicles['charge ac']

RESOLUTION = {1440: 'min', 96: '15min', 24: 'h'}

class Car:

    def __init__(self, random: np.random.default_rng, car_type: str = 'ev', maximal_distance: float = 350, charging_limit: str = 'required',
                 T: int = 1440):
        self.type = car_type
        self.random = random
        # -> select car depending on type and distance
        if self.type == 'ev':
            properties = self.get_electric_vehicle(maximal_distance)
        else:
            properties = dict(model='Nowum Car', capacity=40, consumption=7, distance=40/7 * 100,
                maximal_charging_power=None)

        # -> time resolution information
        self.T, self.t, self.dt = T, np.arange(T), 1/(T/24)        
        # -> technical parameters
        self.model = properties['model']                                        # -> model type
        self.capacity = properties['capacity']                                  # -> capacity [kWh]
        self.distance = round(properties['distance'], 2)                        # -> maximal distance [km]
        self.consumption = properties['consumption'] / 100                      # -> consumption [kWh/km]
        self.maximal_charging_power = properties['maximal_charging_power']      # -> rated power [kW]
        self.soc = self.random.integers(low=10, high=90)/100                         # -> state of charge [0,..., 1]
        self.odometer = 0                                                       # -> distance counter
        # -> charging parameters
        self.charging_limit = charging_limit                                    # -> default strategy
        self.daily_limit = {}                                                   # -> limit @ day

        self._data = pd.DataFrame(columns=['distance', 'total_distance', 'soc', 'planned_charge', 'final_charge',
                                           'demand', 'usage', 'work', 'errand', 'hobby'])
        # -> simulation monitoring
        self.empty = False                                                      # -> True if car has not enough energy
        self.virtual_source = 0                                                 # -> used energy if car is empty

    # -> function to get the EV for the corresponding distance
    def get_electric_vehicle(self, distance):
        possible_vehicles = electric_vehicles.loc[electric_vehicles['distance'] > distance]
        if len(possible_vehicles) > 0:
            probabilities = (1/possible_vehicles['weight'].sum() * possible_vehicles['weight']).values
            index = self.random.choice(possible_vehicles.index, p=probabilities)
            vehicle = possible_vehicles.loc[index].to_dict()
        else:
            vehicle = electric_vehicles.iloc[electric_vehicles['distance'].idxmax()].to_dict()

        return vehicle

    def initialize_time_series(self, mobility: MobilityDemand, start_date: datetime, end_date: datetime):

        def round_ts_to_base(ts: datetime, b: int):
            if b * round((ts.minute / b)) == 60:
                ts = ts + td(hours=1)
                ts = ts.replace(minute=0)
            else:
                ts = ts.replace(minute=base * round((ts.minute / base)))

            return ts

        # -> initialize time stamps
        time_range = pd.date_range(start=start_date, end=end_date + td(days=1), freq=RESOLUTION[self.T])[:-1]
        date_range = pd.date_range(start=start_date, end=end_date, freq='d')

        base = int(60 / (self.T / 24))

        # -> initialize time series
        for column in self._data.columns:
            self._data[column] = np.zeros(len(time_range))
        self._data.index = time_range

        # -> for each day get car usage
        for key, mobilities in mobility.mobility.items():
            days = date_range[date_range.day_name() == key]
            demand_ = 0
            for mobility in mobilities:
                # -> demand in [kWh]
                demand = mobility['distance'] * self.consumption
                demand_ += 2 * (mobility['distance'] * self.consumption)

                travel_time = max(mobility['travel_time'], base)
                # -> departure time
                t1 = datetime.strptime(mobility['start_time'], '%H:%M:%S')
                # -> arrival time
                t2 = t1 + td(minutes=mobility['duration'] + travel_time)

                # -> set demand for mondays, tuesdays, ...
                for day in days:
                    if t2.day > t1.day:
                        arrival = day.combine(day + td(days=1), t2.time())
                    else:
                        arrival = day.combine(day, t2.time())
                    # -> set time series
                    destination = day.combine(day, t1.time())
                    destination = round_ts_to_base(destination, base)

                    departure = round_ts_to_base((destination - td(minutes=travel_time)), base)
                    steps = len(self._data.loc[departure:destination, 'demand'])
                    if departure >= start_date and destination <= end_date:
                        self._data.loc[departure:destination, 'demand'] = demand / steps / self.dt

                    destination = round_ts_to_base((arrival - td(minutes=travel_time)), base)
                    if destination >= start_date and arrival <= end_date:
                        self._data.loc[destination:arrival, 'demand'] = demand / steps / self.dt
                        self._data.loc[departure:arrival, 'usage'] = 1
                        self._data.loc[departure:arrival, mobility['type']] = 1
                    elif destination < end_date:
                        self._data.loc[destination:time_range[-1], 'demand'] = 1
                        self._data.loc[departure:time_range[-1], mobility['type']] = 1

            if len(days) > 0:
                if self.charging_limit == 'max':
                    self.daily_limit[days[0].weekday()] = 0.98
                elif self.charging_limit == 'required':
                    self.daily_limit[days[0].weekday()] = demand_ / self.capacity
                else:
                    self.daily_limit[days[0].weekday()] = 0.5

    def drive(self, d_time: datetime):
        # -> if no demand is set --> demand 1 % of Soc
        demand = self._data.loc[d_time, 'demand'] * self.dt
        distance = demand / self.consumption
        self.odometer += distance
        capacity = (self.capacity * self.soc) - demand
        soc = capacity / self.capacity

        # -> check if demand > current capacity
        if soc < 0:
            self.virtual_source += demand
            self.empty, self.soc = True, 0
        else:
            self.empty, self.soc = False, soc

        self._data.at[d_time, 'soc'] = self.soc
        self._data.at[d_time, 'distance'] = distance
        self._data.at[d_time, 'total_distance'] = self.odometer

        return self._data.loc[d_time, 'demand']

    def charge(self, d_time: datetime):

        # -> charge battery for 1 minute
        capacity = self.capacity * self.soc + self._data.loc[d_time, 'final_charge'] * self.dt
        self.soc = capacity / self.capacity
        self.soc = min(self.soc, 1)

        return self._data.loc[d_time, 'final_charge']

    def get_data(self, column: str) -> pd.Series:
        return self._data[column]

    def get_result(self, time_range: pd.DatetimeIndex = None) -> pd.DataFrame:
        return self._data.loc[time_range]

    def get_limit(self, d_time: datetime, strategy: str = 'required') -> float:
        if strategy == 'max':
            limit = 0.98
        elif strategy == 'required':
            today, l_today = d_time.weekday(), self.soc
            tomorrow, l_tomorrow = (d_time + td(days=1)).weekday(), self.soc
            if today in self.daily_limit.keys():
                l_today, l_tomorrow = self.daily_limit[today], self.daily_limit[today]
            if tomorrow in self.daily_limit.keys():
                l_tomorrow = self.daily_limit[tomorrow]
            t = (d_time.hour * 60 + d_time.minute) / 1440
            limit = (l_tomorrow-l_today) * t + l_today
        else:
            limit = 0.5

        return limit

    def set_planned_charging(self, time_series: pd.Series) -> None:
        self._data.loc[time_series.index, 'planned_charge'] = time_series.values

    def set_final_charging(self, time_series: pd.Series) -> None:
        self._data.loc[time_series.index, 'final_charge'] = time_series.values

    def get_current_capacity(self):
        return self.soc * self.capacity

