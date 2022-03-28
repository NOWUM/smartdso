import numpy as np
import pandas as pd
from datetime import datetime, timedelta as td
from mobLib.mobility_demand import MobilityDemand
from carLib.car import Car

# ---> price data from survey
mean_price = 28.01
var_price = 7.9

power_price = 24


def get_mobility_types():
    employee = np.random.choice(a=[True, False], p=[0.7, 0.3])
    if employee:
        return ['work', 'hobby', 'errand']
    else:
        return ['hobby', 'errand']


class Resident:

    def __init__(self, **kwargs):

        self.mobility = MobilityDemand(get_mobility_types())                            # ---> create mobility pattern

        if self.mobility.car_usage:
            max_distance = self.mobility.maximal_distance
            if max_distance > 600:
                car_type = 'fv'
            else:
                car_type = np.random.choice(a=['ev', 'fv'], p=[kwargs['ev_ratio'], 1 - kwargs['ev_ratio']])

            self.car = Car(car_type=car_type, maximal_distance=max_distance, charging_limit=kwargs['minimum_soc'])
            self.car.set_demand(self.mobility, kwargs['start_date'], kwargs['end_date'])

        else:
            self.car = Car(car_type='no car')

        # ---> price limits from survey
        self.price_low = round(np.random.normal(loc=mean_price, scale=var_price), 2)    # ---> charge
        self.price_medium = 0.805 * self.price_low + 17.45                              # ---> require
        self.price_limit = 1.1477 * self.price_medium + 1.51 - power_price              # ---> reject
        self.price_limit = max(self.price_limit, 3)


if __name__ == "__main__":
    sim_paras = dict(start_date=pd.to_datetime('2022-01-01'), end_date=pd.to_datetime('2022-02-01'),
                     ev_ratio=1, minimum_soc=30)
    person = Resident(**sim_paras)
