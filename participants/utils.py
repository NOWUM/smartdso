import numpy as np
import pandas as pd
from mobLib.mobility_demand import MobilityDemand
from carLib.car import Car
from datetime import datetime

# -> price data from survey
MEAN_PRICE = 28.01
VAR_PRICE = 7.9


class Resident:

    def __init__(self, ev_ratio: float, minimum_soc: int, start_date: datetime, end_time: datetime,
                 base_price: float, *args, **kwargs):

        categories = ['hobby', 'errand']
        if np.random.choice(a=[True, False], p=[0.7, 0.3]):
            categories.insert(0, 'work')

        # -> create mobility pattern
        self.mobility = MobilityDemand(categories)
        # -> select car if used
        if self.mobility.car_usage:
            max_distance = self.mobility.maximal_distance
            car_type = 'fv'
            if max_distance < 600:
                car_type = np.random.choice(a=['ev', 'fv'], p=[ev_ratio, 1 - ev_ratio])
            self.car = Car(car_type=car_type, maximal_distance=max_distance, charging_limit=minimum_soc)
            self.car.initialize_time_series(self.mobility, start_date, end_time)
        else:
            self.car = Car(car_type='no car')

        # -> price limits from survey
        prc_level = round(np.random.normal(loc=MEAN_PRICE, scale=VAR_PRICE), 2)
        self.price_limit = max(1.1477 * (0.805 * prc_level + 17.45) + 1.51 - base_price, 5)


if __name__ == "__main__":
    sim_paras = dict(start_date=pd.to_datetime('2022-01-01'), end_date=pd.to_datetime('2022-02-01'),
                     ev_ratio=1, minimum_soc=30)
    person = Resident(**sim_paras)
