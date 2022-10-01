from participants.residential import HouseholdModel
import numpy as np
import pandas as pd
from agents.utils import WeatherGenerator
from datetime import timedelta as td


def run_household_model(strategy: str = 'PlugInCap'):

    start_date = pd.to_datetime('2022-01-01')
    end_date = pd.to_datetime('2022-01-10')

    pv_system = dict(pdc0=5, surface_tilt=35, surface_azimuth=180)

    random = np.random.default_rng(2)

    house = HouseholdModel(residents=1, demandP=5000, pv_systems=[pv_system], random=random,
                           start_date=start_date, end_date=end_date, ev_ratio=1, T=96,
                           strategy=strategy)

    weather_generator = WeatherGenerator()
    weather = pd.concat([weather_generator.get_weather(date=date)
                         for date in pd.date_range(start=start_date, end=end_date + td(days=1),
                                                   freq='d')])
    weather = weather.resample('15min').ffill()
    weather = weather.loc[weather.index.isin(house.time_range)]
    house.set_parameter(weather=weather)

    for t in house.time_range:
        request = house.get_request(d_time=t, strategy=strategy)
        if request.sum() > 0:
            print(f'send request at {t}')
            house.commit(pd.Series(data=np.zeros(len(request)), index=request.index))
        house.simulate(t)

    return house.get_result()
