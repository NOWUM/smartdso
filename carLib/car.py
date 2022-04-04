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

        if self.type == 'ev':
            properties = get_electric_vehicle(maximal_distance)
        else:
            properties = get_fossil_vehicle()

        self.model = properties['model']                                        # ---> model type
        self.capacity = properties['capacity']                                  # ---> capacity [kWh]
        self.distance = round(properties['distance'], 2)                        # ---> maximal distance [km]
        self.consumption = properties['consumption']                            # ---> consumption [kWh/100km]
        self.maximal_charging_power = properties['maximal_charging_power']      # ---> rated power [kW]
        self.soc = np.random.randint(low=10, high=90)                           # ---> state of charge [0,..., 100]
        self.odometer = 0                                                       # ---> distance counter

        self.charging = False
        self.charging_duration = 0
        self.default_limit = charging_limit
        self.limit = charging_limit
        self.charge_anyway = 5

        self.demand = None                                                      # ---> driving demand time series
        self.usage = None                                                       # ---> car home and chargeable

    def set_demand(self, mobility: MobilityDemand, start_date: datetime, end_date: datetime):
        time_range = pd.date_range(start=start_date, end=end_date + td(days=1), freq='min')[:-1]
        date_range = pd.date_range(start=start_date, end=end_date, freq='d')
        self.usage = pd.Series(index=time_range, data=np.zeros(len(time_range)))
        self.demand = pd.Series(index=time_range, data=np.zeros(len(time_range)))
        # ---> for each day calculate car usage
        for key, demands in mobility.mobility.items():
            days = date_range[date_range.day_name() == key]
            for demand in demands:
                mean_consumption = (demand['distance'] * self.consumption / 100) / demand['travel_time']
                t1 = datetime.strptime(demand['start_time'], '%H:%M:%S')
                t_departure = t1 - td(minutes=demand['travel_time'])
                t2 = t1 + td(minutes=demand['duration'])
                t_arrival = t2 + td(minutes=demand['travel_time'])
                for day in days:
                    departure = day.combine(day, t_departure.time())
                    if t2.day > t1.day:
                        arrival = day.combine(day + td(days=1), t_arrival.time())
                    else:
                        arrival = day.combine(day, t_arrival.time())
                    self.usage[departure:arrival] = 1

                    journey = day.combine(day, t1.time())
                    self.demand[departure:journey-td(minutes=1)] = mean_consumption
                    journey = arrival - td(minutes=demand['travel_time'])
                    self.demand[journey:arrival - td(minutes=1)] = mean_consumption

    def drive(self, d_time: datetime):
        if self.demand is None:                                                 # ---> if no demand is st
            self.soc -= 1                                                       # ---> decrease the soc 1 %
            self.soc = max(0, self.soc)
            return self.capacity * 0.01
        else:
            self.odometer += self.demand[d_time] / self.consumption * 100 # ---> use time series
            energy = (self.capacity * self.soc/100) - self.demand[d_time]
            self.soc = max(0, energy / self.capacity * 100)
            return self.demand[d_time]

    def charge(self):
        # ---> charge battery for 1 minute
        if self.charging:
            capacity = self.capacity * self.soc / 100 + self.maximal_charging_power / 60
            self.soc = capacity / self.capacity * 100
            self.soc = min(self.soc, 100)
            self.charging_duration -= 1
            if self.charging_duration == 0:
                self.charging = False
            return self.maximal_charging_power / 60
        else:
            return 0

    def plan_charging(self, d_time: datetime):
        if self.soc < self.limit and not self.charging and self.usage[d_time] == 0:
            chargeable = self.usage.loc[(self.usage == 0) & (self.usage.index >= d_time)]
            car_in_use = self.usage.loc[(self.usage == 1) & (self.usage.index >= d_time)]
            if len(car_in_use) > 0:
                start_time = car_in_use.index[0]
                next_chargeable = self.usage.loc[(self.usage == 0) & (self.usage.index >= start_time)]
                if len(next_chargeable) > 0:
                    end_time = next_chargeable.index[0]
                    total_demand = (self.demand.loc[start_time:end_time].sum() / self.capacity) * 100
                    self.charge_anyway = max(5, round(total_demand), 2)
                    self.limit = max(self.default_limit, self.charge_anyway)
            else:
                self.charge_anyway = 5
                self.limit = self.default_limit
            if len(chargeable) > 0 and len(car_in_use) > 0:
                total_energy = self.capacity - (self.capacity * self.soc) / 100
                duration = int(total_energy / self.maximal_charging_power * 60)
                maximal_duration = int((car_in_use.index[0] - chargeable.index[0]).total_seconds() / 60)
                self.charging_duration = min(maximal_duration, duration)
                return self.maximal_charging_power, self.charging_duration
            return 0, 0
        else:
            return 0, 0


if __name__ == "__main__":
    car_electric = Car()
