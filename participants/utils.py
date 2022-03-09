import os
import numpy as np
import pandas as pd
import names
from datetime import datetime, timedelta as td
from mobLib.mobility_demand import MobilityDemand
from carLib.car import Car

# ---> electric car parameters
ev_ratio = int(os.getenv('EV_RATIO', 80))
ev_ratio = int(ev_ratio)/100
minimum_soc = int(os.getenv('MINIMUM_SOC', 30))
# ---> simulation range
start_date = pd.to_datetime(os.getenv('START_DATE', '2022-02-01'))
end_date = pd.to_datetime(os.getenv('END_DATE', '2022-02-10'))


class Resident:

    def __init__(self, mobility_types: list, type_: str = 'adult', last_name: str = 'nowum'):
        # ---> create first name to identify the resident
        self.name = f'{names.get_first_name(gender=np.random.choice(["male", "female"], p=[0.5, 0.5]))} {last_name}'
        # ---> draw employee status
        self.employee = True if 'work' in mobility_types else False
        # ---> set type (adult, others)
        self.type = type_
        # ---> create mobility generator
        self.mobility_generator = MobilityDemand(mobility_types)
        # ---> is a car used
        if self.mobility_generator.car_usage:
            self.own_car = True
        else:
            self.own_car = False

        self.car = Car(type=np.random.choice(a=['ev', 'fv'], p=[ev_ratio, 1-ev_ratio])) if self.own_car else None

        time_range = pd.date_range(start=start_date, end=end_date, freq='min')
        self.car_usage = pd.Series(index=time_range, data=np.zeros(len(time_range)))
        self.car_demand = pd.Series(index=time_range, data=np.zeros(len(time_range)))
        if self.own_car and self.car.type == 'ev':
            self._set_car_usage()
            self.car.demand = self.car_demand

    def _set_car_usage(self):
        # ---> for each day calculate car usage
        for day in pd.date_range(start=start_date, end=end_date, freq='d'):
            mobility_demands = self.mobility_generator.mobility[day.date().strftime("%A")]
            for demand in [demand for demand in mobility_demands if demand['car_use']]:
                # ---> determine departure and arrival times
                t1 = datetime.strptime(demand['start_time'], '%H:%M:%S')
                departure = t1 - td(minutes=demand['travel_time'])
                departure = day.replace(hour=departure.hour, minute=departure.minute)
                t1 = departure + td(minutes=demand['travel_time'])
                arrival = departure + td(minutes=demand['duration'] + 2 * demand['travel_time'])
                t2 = arrival - td(minutes=demand['travel_time'])
                # ---> car and resident are not @ home
                for t in pd.date_range(start=departure, end=arrival, freq='min'):
                    self.car_usage.loc[t] = 1
                # ---> set consumption
                mean_consumption = (demand['distance'] * self.car.consumption / 100) / demand['travel_time']
                for t in pd.date_range(start=departure, end=t1, freq='min'):
                    self.car_demand.loc[t] = mean_consumption
                for t in pd.date_range(start=arrival, end=t2, freq='min'):
                    self.car_demand.loc[t] = mean_consumption

    def plan_charging(self, d_time: datetime):
        chargeable = self.car_usage.loc[(self.car_usage == 0) & (self.car_usage.index >= d_time)]
        car_in_use = self.car_usage.loc[(self.car_usage == 1) & (self.car_usage.index >= d_time)]

        if self.car.soc < minimum_soc:
            total_energy = self.car.capacity - (self.car.capacity * self.car.soc) / 100
            duration = int(total_energy / self.car.maximal_charging_power * 60)
            maximal_duration = (car_in_use.index[0] - chargeable.index[0]).total_seconds() / 60
            if duration > maximal_duration:
                time_range = pd.date_range(start=chargeable.index[0], end=car_in_use.index[0], freq='min')
            else:
                time_range = pd.date_range(start=chargeable.index[0], end=chargeable.index[0] + td(minutes=duration),
                                           freq='min')
            return pd.Series(data=np.ones(len(time_range)) * self.car.maximal_charging_power, index=time_range)
        else:
            return pd.Series(dtype=float)


if __name__ == "__main__":
    res = Resident(mobility_types=['work', 'hobby', 'errands'], type_='adult')
    res.car.soc = 80
    res.plan_charging(start_date)
    print(res.mobility_generator.mobility)
    for t in pd.date_range(start=start_date, end=end_date, freq='min'):
        print(t)
        print(res.car_demand[t])
        res.car.drive(t)
