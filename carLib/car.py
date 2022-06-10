from datetime import datetime, timedelta as td
import pandas as pd
import numpy as np

from mobLib.mobility_demand import MobilityDemand

# -> load electric vehicle data
electric_vehicles = pd.read_csv(r'./carLib/data/evs.csv', sep=';', decimal=',')
electric_vehicles['maximal_charging_power'] = electric_vehicles['charge ac']


# -> function to get the EV for the corresponding distance
def get_electric_vehicle(distance):
    possible_vehicles = electric_vehicles.loc[electric_vehicles['distance'] > distance]
    if len(possible_vehicles) > 0:
        probabilities = (1/possible_vehicles['weight'].sum() * possible_vehicles['weight']).values
        index = np.random.choice(possible_vehicles.index, p=probabilities)
        vehicle = possible_vehicles.loc[index].to_dict()
    else:
        vehicle = electric_vehicles.iloc[electric_vehicles['distance'].idxmax()].to_dict()

    return vehicle


def get_fossil_vehicle():
    return dict(model='Nowum Car', capacity=40, consumption=7, distance=40/7 * 100,
                maximal_charging_power=None)


class Car:

    def __init__(self, car_type: str = 'ev', maximal_distance: float = 350, charging_limit: str = 'required'):
        self.type = car_type
        # -> select car depending on type and distance
        if self.type == 'ev':
            properties = get_electric_vehicle(maximal_distance)
        else:
            properties = get_fossil_vehicle()

        # -> technical parameters
        self.model = properties['model']                                        # -> model type
        self.capacity = properties['capacity']                                  # -> capacity [kWh]
        self.distance = round(properties['distance'], 2)                        # -> maximal distance [km]
        self.consumption = properties['consumption'] / 100                      # -> consumption [kWh/km]
        self.maximal_charging_power = properties['maximal_charging_power']      # -> rated power [kW]
        self.soc = np.random.randint(low=10, high=20)/100                       # -> state of charge [0,..., 1]
        self.odometer = 0                                                       # -> distance counter
        # -> charging parameters
        self.charging_limit = charging_limit                                    # -> default strategy
        self.daily_limit = {}                                                   # -> limit @ day
        self.charging = None
        # -> demand parameters
        self.demand = pd.Series(dtype=float)                                    # -> driving demand time series
        self.usage = pd.Series(dtype=float)                                     # -> car @ home and chargeable
        self.monitor = pd.DataFrame(columns=['distance', 'odometer', 'soc',
                                             'work', 'errand', 'hobby'])        # -> use car for job, errand or hobby
        # -> simulation monitoring
        self.empty = False                                                      # -> True if car has not enough energy
        self.virtual_source = 0                                                 # -> used energy if car is empty

    def initialize_time_series(self, mobility: MobilityDemand, start_date: datetime, end_date: datetime):
        # -> initialize time stamps
        time_range = pd.date_range(start=start_date, end=end_date + td(days=1), freq='min')[:-1]
        date_range = pd.date_range(start=start_date, end=end_date, freq='d')
        # -> initialize time series
        self.usage = pd.Series(index=time_range, data=np.zeros(len(time_range)))
        self.demand = pd.Series(index=time_range, data=np.zeros(len(time_range)))
        self.charging = pd.Series(index=time_range, data=np.zeros(len(time_range)))
        for column in self.monitor.columns:
            self.monitor[column] = np.zeros(len(time_range))
        self.monitor.index = time_range

        # -> for each day get car usage
        for key, mobilities in mobility.mobility.items():
            days = date_range[date_range.day_name() == key]
            demand_ = 0
            for mobility in mobilities:
                # -> demand in [kWh/min]
                demand = (mobility['distance'] * self.consumption) / mobility['travel_time']
                demand_ += 2 * (mobility['distance'] * self.consumption)
                # -> departure time
                t1 = datetime.strptime(mobility['start_time'], '%H:%M:%S')
                t_departure = t1 - td(minutes=mobility['travel_time'])
                # -> arrival time
                t2 = t1 + td(minutes=mobility['duration'])
                t_arrival = t2 + td(minutes=mobility['travel_time'])
                # -> set demand for mondays, tuesdays, ...
                for day in days:
                    departure = day.combine(day, t_departure.time())
                    if t2.day > t1.day:
                        arrival = day.combine(day + td(days=1), t_arrival.time())
                    else:
                        arrival = day.combine(day, t_arrival.time())
                    # -> set time series
                    self.usage[departure:arrival] = 1
                    self.monitor.loc[departure:arrival, mobility['type']] = 1
                    destination = day.combine(day, t1.time())-td(minutes=1)
                    self.demand[departure:destination] = demand
                    destination = arrival - td(minutes=mobility['travel_time'] - 1)
                    self.demand[destination:arrival] = demand

            if len(days) > 0:
                if self.charging_limit == 'max':
                    self.daily_limit[days[0].weekday()] = 0.98
                elif self.charging_limit == 'required':
                    self.daily_limit[days[0].weekday()] = demand_ / self.capacity
                else:
                    self.daily_limit[days[0].weekday()] = 0.5

    def drive(self, d_time: datetime):
        # -> if no demand is set --> demand 1 % of Soc
        if self.demand is None:
            self.soc -= 0.01
            self.soc = max(0, self.soc)
            return self.capacity * 0.01
        # -> else use demand time series
        else:
            demand = self.demand.loc[d_time]
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

        self.monitor.at[d_time, 'soc'] = self.soc
        self.monitor.at[d_time, 'distance'] = distance
        self.monitor.at[d_time, 'odometer'] = self.odometer

        return self.demand.loc[d_time]

    def charge(self, d_time: datetime):
        # -> if no charging set = charge 1 % of Soc
        if self.charging is None:
            self.soc += 0.01
            self.soc = min(self.soc, 1)
            return self.capacity * 0.01
        else:
            # -> charge battery for 1 minute
            capacity = self.capacity * self.soc + self.charging.loc[d_time] / 60
            self.soc = capacity / self.capacity
            self.soc = min(self.soc, 1)

        return self.charging.loc[d_time]

    def get_limit(self, d_time: datetime, strategy: str = 'required'):
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
