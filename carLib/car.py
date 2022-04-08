from datetime import datetime, timedelta as td
import pandas as pd
import numpy as np

from mobLib.mobility_demand import MobilityDemand

# ---> load electric vehicle data
electric_vehicles = pd.read_csv(r'./carLib/data/evs.csv', sep=';', decimal=',')
electric_vehicles['maximal_charging_power'] = electric_vehicles['charge ac']


# ---> function to get the EV for the corresponding distance
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

    def __init__(self, car_type: str = 'ev', maximal_distance: float = 350, charging_limit: int = 30):
        self.type = car_type
        # --> select car depending on type and distance
        if self.type == 'ev':
            properties = get_electric_vehicle(maximal_distance)
        else:
            properties = get_fossil_vehicle()

        # --> technical parameters
        self.model = properties['model']                                        # --> model type
        self.capacity = properties['capacity']                                  # --> capacity [kWh]
        self.distance = round(properties['distance'], 2)                        # --> maximal distance [km]
        self.consumption = properties['consumption'] / 100                      # --> consumption [kWh/km]
        self.maximal_charging_power = properties['maximal_charging_power']      # --> rated power [kW]
        self.soc = np.random.randint(low=60, high=80)                           # --> state of charge [0,..., 100]
        self.odometer = 0                                                       # --> distance counter
        # --> charging parameters
        self.charging = False                                                   # --> true if car charges
        self.duration = 0                                                       # --> charging duration
        self.limit = charging_limit                                             # --> default limit
        self.daily_limit = {}                                                   # --> limit @ day
        self.require = 5                                                        # --> lower bound (price not matter)
        # --> demand parameters
        self.demand = None                                                      # --> driving demand time series
        self.usage = None                                                       # --> car @ home and chargeable
        # --> simulation monitoring
        self.empty = False                                                      # --> True if car has not enough energy
        self.virtual_source = 0                                                 # --> used energy if car is empty

    def set_demand(self, mobility: MobilityDemand, start_date: datetime, end_date: datetime):
        # --> initialize time series
        time_range = pd.date_range(start=start_date, end=end_date + td(days=1), freq='min')[:-1]
        date_range = pd.date_range(start=start_date, end=end_date, freq='d')
        self.usage = pd.Series(index=time_range, data=np.zeros(len(time_range)))
        self.demand = pd.Series(index=time_range, data=np.zeros(len(time_range)))
        # --> for each day get car usage
        for key, mobilities in mobility.mobility.items():
            days = date_range[date_range.day_name() == key]
            demand_ = 0.
            for mobility in mobilities:
                # --> demand in [kWh/min]
                demand = (mobility['distance'] * self.consumption) / mobility['travel_time']
                demand_ += 2 * (mobility['distance'] * self.consumption)
                # --> departure time
                t1 = datetime.strptime(mobility['start_time'], '%H:%M:%S')
                t_departure = t1 - td(minutes=mobility['travel_time'])
                # --> arrival time
                t2 = t1 + td(minutes=mobility['duration'])
                t_arrival = t2 + td(minutes=mobility['travel_time'])
                # --> set demand for mondays, tuesdays, ...
                for day in days:
                    departure = day.combine(day, t_departure.time())
                    if t2.day > t1.day:
                        arrival = day.combine(day + td(days=1), t_arrival.time())
                    else:
                        arrival = day.combine(day, t_arrival.time())
                    # --> set time series
                    self.usage[departure:arrival] = 1
                    destination = day.combine(day, t1.time())-td(minutes=1)
                    self.demand[departure:destination] = demand
                    destination = arrival - td(minutes=mobility['travel_time'] - 1)
                    self.demand[destination:arrival] = demand
            if len(days) > 0:
                index = days[0].weekday()
                if self.limit == 100:
                    self.daily_limit[index] = 98
                elif self.limit == -1:
                    self.daily_limit[index] = round(demand_ / self.capacity * 100, 2)
                else:
                    self.daily_limit[index] = self.limit

    def drive(self, d_time: datetime):
        if self.demand is None:                 # --> no demand set = demand 1% Soc
            self.soc -= 1
            self.soc = max(0, self.soc)
            return self.capacity * 0.01

        else:                                   # --> use demand time series
            demand = self.demand[d_time]
            self.odometer += demand / self.consumption
            capacity = (self.capacity * self.soc/100) - self.demand[d_time]
            soc = capacity / self.capacity * 100
        # --> check if demand > current capacity
        if soc < 0:
            self.virtual_source += self.demand[d_time]
            self.empty, self.soc = True, 0
        else:
            self.empty, self.soc = False, soc

        return self.demand[d_time]

    def charge(self):
        # ---> charge battery for 1 minute
        if self.charging:
            capacity = self.capacity * self.soc / 100 + self.maximal_charging_power / 60
            self.soc = capacity / self.capacity * 100
            self.soc = min(self.soc, 100)
            self.duration -= 1
            if self.duration == 0:
                self.charging = False
            return self.maximal_charging_power / 60
        else:
            return 0

    def plan_charging(self, d_time: datetime):

        if self.limit == 100:
            limit = 98
        elif self.limit == -1:
            today, l_today = d_time.weekday(), 0
            tomorrow, l_tomorrow = (d_time + td(days=1)).weekday(), 0
            if today in self.daily_limit.keys():
                l_today, l_tomorrow = self.daily_limit[today], self.daily_limit[today]
            if tomorrow in self.daily_limit.keys():
                l_tomorrow = self.daily_limit[tomorrow]
            if d_time.hour >= 18:
                t = ((d_time.hour - 18) * 60 + d_time.minute) / 360
                limit = (l_tomorrow-l_today) * t + l_today
            else:
                limit = l_today
        else:
            limit = self.limit

        if self.soc < limit and not self.charging and self.usage[d_time] == 0:
            chargeable = self.usage.loc[(self.usage == 0) & (self.usage.index >= d_time)]
            car_in_use = self.usage.loc[(self.usage == 1) & (self.usage.index >= d_time)]
            if len(car_in_use) > 0:
                start_time = car_in_use.index[0]
                next_chargeable = self.usage.loc[(self.usage == 0) & (self.usage.index >= start_time)]
                if len(next_chargeable) > 0:
                    end_time = next_chargeable.index[0]
                    min_soc = 1.1 * (self.demand.loc[start_time:end_time].sum() / self.capacity) * 100
                    self.require = max(5, min_soc)

            if len(chargeable) > 0 and len(car_in_use) > 0:
                total_energy = self.capacity - (self.capacity * self.soc) / 100
                duration = int(total_energy / self.maximal_charging_power * 60)
                maximal_duration = int((car_in_use.index[0] - chargeable.index[0]).total_seconds() / 60)
                self.duration = min(maximal_duration, duration)
                return self.maximal_charging_power, self.duration
            return 0, 0
        else:
            return 0, 0


if __name__ == "__main__":
    car_electric = Car()
