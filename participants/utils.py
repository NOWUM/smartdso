import numpy as np
import pandas as pd
from mobLib.mobility_demand import MobilityDemand
from carLib.car import Car
from datetime import datetime


class Resident:

    def __init__(self, ev_ratio: float, start_date: datetime, end_time: datetime, T: int = 1440,
                 charging_limit: str = 'required', *args, **kwargs):

        self.T, self.t, self.dt = T, np.arange(T), 1/(T/24)

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
            self.car = Car(car_type=car_type, maximal_distance=max_distance, charging_limit=charging_limit, T=self.T)
            self.car.initialize_time_series(self.mobility, start_date, end_time)
        else:
            self.car = Car(car_type='no car')


if __name__ == "__main__":
    sim_paras = dict(start_date=pd.to_datetime('2022-01-01'), end_date=pd.to_datetime('2022-02-01'),
                     ev_ratio=1, minimum_soc=30)
    person = Resident(**sim_paras)


    # def _plan_without_photovoltaic(self, d_time: datetime, strategy: str):
    #     cars = [person.car for person in self.persons if person.car.type == 'ev']
    #
    #     remaining_steps = len(self.time_range[self.time_range >= d_time])
    #     self.request = pd.Series(data=np.zeros(remaining_steps),
    #                              index=pd.date_range(start=d_time, freq=RESOLUTION[self.T], periods=remaining_steps))
    #     for car, c in zip(cars, range(len(cars))):
    #         self.charging[c] = pd.Series(data=np.zeros(remaining_steps),
    #                                      index=pd.date_range(start=d_time, freq=RESOLUTION[self.T],
    #                                                          periods=remaining_steps))
    #         # -> plan charge if the soc < limit and the car is not already charging and the car is at home
    #         if car.soc < car.get_limit(d_time, strategy) and car.usage[d_time] == 0:
    #             logger.debug(f'plan charging without photovoltaic for car: {c}')
    #             # -> get first time stamp of next charging block
    #             chargeable = car.usage.loc[(car.usage == 0) & (car.usage.index >= d_time)]
    #             if chargeable.empty:
    #                 t1 = self.time_range[-1]
    #             else:
    #                 t1 = chargeable.index[0]
    #             # -> get first time stamp of next using block
    #             car_in_use = car.usage.loc[(car.usage == 1) & (car.usage.index >= d_time)]
    #             if car_in_use.empty:
    #                 t2 = self.time_range[-1]
    #             else:
    #                 t2 = car_in_use.index[0]
    #             # -> if d_time in charging block --> plan charging
    #             if t2 > t1:
    #                 total_energy = car.capacity - car.capacity * car.soc
    #                 limit_by_capacity = round(total_energy / car.maximal_charging_power * 1/self.dt)
    #                 limit_by_slot = len(self.time_range[(self.time_range >= t1) & (self.time_range <= t2)])
    #                 duration = int(min(limit_by_slot, limit_by_capacity))
    #                 self.charging[c] = pd.Series(data=car.maximal_charging_power * np.ones(duration),
    #                                              index=pd.date_range(start=d_time, freq=RESOLUTION[self.T],
    #                                                                  periods=duration))
    #                 # -> add planed charging to power
    #                 self.request.loc[self.charging[c].index] += self.charging[c].values
    #
    #             self.request = self.request.loc[self.request > 0]