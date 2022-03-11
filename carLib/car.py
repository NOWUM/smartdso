import datetime
from datetime import datetime

import pandas as pd
import numpy as np

electric_vehicles = pd.read_csv(r'./carLib/data/top_10_evs.csv')
electric_vehicles['distance'] = electric_vehicles['capacity']/electric_vehicles['consumption'] * 100
electric_vehicles['maximal_charging_power'] = 22


def get_electric_vehicle(distance):

    possible_vehicles = electric_vehicles.loc[electric_vehicles['distance'] > distance]
    if len(possible_vehicles) > 0:
        probabilities = (1/possible_vehicles['weight'].sum() * possible_vehicles['weight']).values
        index = np.random.choice(possible_vehicles.index, p=probabilities)
    else:
        possible_vehicles = electric_vehicles.loc[electric_vehicles['distance'] == electric_vehicles['distance'].max()]
        index = 0

    return possible_vehicles.loc[index, ['model', 'capacity', 'consumption', 'distance',
                                         'maximal_charging_power']].to_dict()


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

        self.model = properties['model']
        self.capacity = properties['capacity']
        self.distance = round(properties['distance'], 2)
        self.consumption = properties['consumption']
        self.maximal_charging_power = properties['maximal_charging_power']
        self.soc = np.random.randint(low=40, high=70)

        self.total_distance = 0
        self.demand = None

    def drive(self, d_time: datetime):
        if self.demand is None:
            self.soc -= 1
        else:
            self.total_distance += self.demand[d_time] / self.consumption * 100
            energy = (self.capacity * self.soc/100) - self.demand[d_time]
            self.soc = round(energy / self.capacity * 100,2)

    def charge(self):
        capacity = self.capacity * self.soc / 100 + self.maximal_charging_power/60
        self.soc = capacity / self.capacity * 100
        self.soc = min(self.soc, 100)


if __name__ == "__main__":
    car_electric = Car()
