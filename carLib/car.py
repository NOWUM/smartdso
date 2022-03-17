from datetime import datetime, timedelta as td
import pandas as pd
import numpy as np

# ---> load electric vehicle data
# electric_vehicles = pd.read_csv(r'./carLib/data/top_10_evs.csv')
# electric_vehicles['distance'] = electric_vehicles['capacity']/electric_vehicles['consumption'] * 100
# electric_vehicles['maximal_charging_power'] = 22
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

    def __init__(self, type: str = 'ev', maximal_distance: float = 350):
        self.type = type

        if self.type == 'ev':
            properties = get_electric_vehicle(maximal_distance)
        else:
            properties = get_fossil_vehicle()

        self.model = properties['model']                                        # ---> model type
        self.capacity = properties['capacity']                                  # ---> capacity [kWh]
        self.distance = round(properties['distance'], 2)                        # ---> maximal distance [km]
        self.consumption = properties['consumption']                            # ---> consumption [kWh/100km]
        self.maximal_charging_power = properties['maximal_charging_power']      # ---> fixed 22 [kW]
        self.soc = np.random.randint(low=20, high=80)                           # ---> state of charge [0,..., 100]
        self.total_distance = 0                                                 # ---> distance counter
        self.demand = None                                                      # ---> driving demand time series

    def drive(self, d_time: datetime):
        if self.demand is None:                                                 # ---> if no demand is st
            self.soc -= 1                                                       # ---> decrease the soc 1 %
            self.soc = max(0, self.soc)
            return self.capacity * 0.01
        else:
            self.total_distance += self.demand[d_time] / self.consumption * 100 # ---> use time series
            energy = (self.capacity * self.soc/100) - self.demand[d_time]
            self.soc = max(0, energy / self.capacity * 100)
            return self.demand[d_time]

    def charge(self):
        # ---> charge battery for 1 minute
        capacity = self.capacity * self.soc / 100 + self.maximal_charging_power/60
        self.soc = capacity / self.capacity * 100
        self.soc = min(self.soc, 100)
        return self.maximal_charging_power / 60


if __name__ == "__main__":
    car_electric = Car()
