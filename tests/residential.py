from participants.residential import HouseholdModel
import numpy as np
import pandas as pd
from agents.utils import WeatherGenerator
from datetime import timedelta as td


def run_household_model(strategy: str = 'PlugInCap', test_commit: bool = True):

    start_date = pd.to_datetime('2022-01-01')
    end_date = pd.to_datetime('2022-01-10')

    pv_system = dict(pdc0=5, surface_tilt=35, surface_azimuth=180)

    random = np.random.default_rng(2)
    if 'Soc' in strategy:
        scenario = 'Spot'
    else:
        scenario = 'Flat'
    house = HouseholdModel(residents=1, demandP=5000, pv_systems=[pv_system], random=random,
                           start_date=start_date, end_date=end_date, ev_ratio=1, T=96,
                           strategy=strategy, scenario=scenario)

    weather_generator = WeatherGenerator()
    weather = pd.concat([weather_generator.get_weather(date=date)
                         for date in pd.date_range(start=start_date, end=end_date + td(days=1),
                                                   freq='d')])
    weather = weather.resample('15min').ffill()
    weather = weather.loc[weather.index.isin(house.time_range)]
    house.set_parameter(weather=weather)
    house.initial_time_series()
    for t in house.time_range:
        if 'MaxPv' in strategy:
            if t.hour == 0 and t.minute == 0:
                request = house.get_request(d_time=t, strategy=strategy)
                if request.sum() > 0:
                    print(f'send request at {t}')
                    print(request)
                    if test_commit:
                        house.commit(pd.Series(data=np.zeros(len(request)), index=request.index))
                    else:
                        res = house.commit(pd.Series(data=20 * np.ones(len(request)), index=request.index))
                        assert not res
            else:
                house.simulate(t)
        else:
            request = house.get_request(d_time=t, strategy=strategy)
            if request.sum() > 0:
                print(f'send request at {t}')
                print(request)
                if test_commit:
                    if house._commit < t:
                        house.commit(pd.Series(data=np.zeros(len(request)), index=request.index))
                else:
                    if house._commit < t:
                        house.commit(pd.Series(data=20 * np.ones(len(request)), index=request.index))
            house.simulate(t)
        for car in house.cars.values():
            if car.soc == 0.1:
                print('SoC:', car.soc, strategy)

    return house.get_result()


def test_household():
    for strategy in ['PlugInCap', 'MaxPvCap', 'MaxPvSoc']:
        print('----------------------------')
        print(f'checking {strategy}')
        print('----------------------------')
        d1 = run_household_model(strategy=strategy)
        assert all(d1['planned_grid_consumption'] == d1['final_grid_consumption'])
        print('----------------------------')
        print('checking commit')
        d2 = run_household_model(strategy=strategy, test_commit=False)
        assert any(d2['planned_grid_consumption'] != d2['final_grid_consumption'])
        print('----------------------------')