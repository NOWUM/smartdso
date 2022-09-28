import numpy as np
from mobLib.mobility_demand import MobilityDemand
from carLib.car import Car
from datetime import datetime
import os

i = 0
def key_generator():
    global i 
    i += 1
    return i 


class Resident:

    def __init__(self, ev_ratio: float, start_date: datetime, end_time: datetime, 
                 random: np.random.default_rng,
                 T: int = 1440,
                 charging_limit: str = 'required',
                 *args, **kwargs):

        self.T, self.t, self.dt = T, np.arange(T), 1/(T/24)

        self.random = random

        categories = ['hobby', 'errand']
        if self.random.choice(a=[True, False], p=[0.7, 0.3]):
            categories.insert(0, 'work')

        # -> create mobility pattern
        self.mobility = MobilityDemand(random, categories)
        # -> select car if used
        if self.mobility.car_usage:
            max_distance = self.mobility.maximal_distance
            car_type = 'fv'
            if max_distance < 600:
                car_type = self.random.choice(a=['ev', 'fv'], p=[ev_ratio, 1 - ev_ratio])
            self.car = Car(self.random, car_type=car_type, maximal_distance=max_distance, charging_limit=charging_limit, T=self.T)
            self.car.initialize_time_series(self.mobility, start_date, end_time)
        else:
            self.car = Car(self.random, car_type='no car')


