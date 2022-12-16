import numpy as np
import pandas as pd
from datetime import timedelta as td

from participants.resident import Resident
from carLib.car import CarData


def sim_one_car(rnd: int = 5):

    start_date = pd.Timestamp(2023, 1, 1)
    end_date = pd.Timestamp(2024, 1, 1)
    time_range = pd.date_range(start=start_date, end=end_date, freq='15min')

    resident = Resident(ev_ratio=1, start_date=start_date, end_date=end_date,
                        random=np.random.default_rng(rnd))

    car = resident.car

    if car.model == 'Nowum Car':
        return np.zeros(len(time_range))

    last_charge = time_range[0] - td(minutes=-1)

    try:
        for t in time_range:
            usage = car.get(CarData.usage, slice(t, None))
            if last_charge < t:
                if car.soc < car.get_limit(t, 'required') and usage.at[t] == 0:
                    chargeable = usage.loc[usage == 0]
                    # -> get first time stamp of next charging block
                    if chargeable.empty:
                        t1 = time_range[-1]
                    else:
                        t1 = chargeable.index[0]
                    # -> get first time stamp of next using block
                    car_in_use = usage.loc[usage == 1]
                    if car_in_use.empty:
                        t2 = time_range[-1]
                    else:
                        t2 = car_in_use.index[0]

                    if t2 > t1:
                        limit_by_capacity = (car.capacity * (1 - car.soc)) / car.maximal_charging_power / 0.25
                        limit_by_slot = len(time_range[(time_range >= t1) & (time_range < t2)]) / 0.25
                        duration = int(min(limit_by_slot, limit_by_capacity))
                        car_power = pd.Series(
                            data=car.maximal_charging_power * np.ones(duration),
                            index=pd.date_range(
                                start=t, freq='15min', periods=duration
                            ),
                        )

                        car.set_final_charging(car_power)
                        last_charge = car_power.loc[car_power > 0].index[-1]

            resident.car.drive(t)
            resident.car.charge(t)

        return car.get(CarData.final_charge, time_range).values

    except Exception as e:
        print(repr(e))
        return np.zeros(len(time_range))