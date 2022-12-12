import numpy as np
import pandas as pd
from collections import deque
from copy import copy

DAY_NAMES = ('Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday')

REFERENCE_TEMPERATURE = np.load(r"./demLib/data/heat_demand/reference_temperature.bin")
idx = pd.date_range(start="2018-01-01", freq="h", periods=len(REFERENCE_TEMPERATURE))
REFERENCE_TEMPERATURE = pd.DataFrame(data=dict(temp_air=REFERENCE_TEMPERATURE), index=idx)

SINK_TEMPERATURE = {'radiator': lambda temp: 40 - 1.0 * temp,
                    'floor': lambda temp: 30 - 0.5 * temp,
                    'hot_water': lambda temp: np.array([50 for _ in temp])}

HOURLY_FACTORS = {}
BUILDING_PARAMETERS = {}
DAILY_FACTORS = {}
for consumer_type in ['EFH', 'MFH', 'GKO', 'GH', 'GMK']:
    # -> get hourly factors
    data_frame = pd.read_excel(r'./demLib/data/heat_demand/hourly_factors.xlsx', sheet_name=consumer_type)
    data_frame = data_frame.fillna(method='ffill')
    if consumer_type in ['EFH', 'MFH']:
        data_frame = data_frame.loc[data_frame['Klasse'] == 'Klasse 3']
        dfs = []
        for day_name in DAY_NAMES:
            data_frame['Klasse'] = day_name
            dfs.append(data_frame.copy())
        data_frame = pd.concat(dfs)
    data_frame = data_frame.set_index(['Klasse', 'Niveau'])
    HOURLY_FACTORS[consumer_type] = data_frame.copy()
    # -> get building parameters
    data_frame = pd.read_excel(r'./demLib/data/heat_demand/building_factors.xlsx', sheet_name='building')
    data_frame = data_frame.set_index('type')
    BUILDING_PARAMETERS = data_frame.to_dict(orient='index')
    data_frame = pd.read_excel(r'./demLib/data/heat_demand/building_factors.xlsx', sheet_name='weekday')
    data_frame = data_frame.set_index('type')
    DAILY_FACTORS = data_frame.to_dict(orient='index')


def get_hourly_factors(mean_temp: float, day: str = 'Monday', consumer_type: str = 'EFH'):
    factors = HOURLY_FACTORS[consumer_type]
    factors = factors.loc[factors.index.get_level_values('Klasse') == day]
    if mean_temp > 0:
        row = min(int(mean_temp / 5) + 4, 9)
    else:
        row = max(int(mean_temp / 5) + 3, 0)

    return factors.iloc[row, :].values.flatten()


def get_cop_time_series(delta_theta: np.array, hp_type: str = 'ASHP'):
    func = {'ASHP': lambda x: 6.01 - 0.09 * x + 0.0005 * x ** 2,
            'GSHP': lambda x: 10.29 - 0.21 * x + 0.0012 * x ** 2,
            'WSHP': lambda x: 9.97 - 0.20 * x + 0.0012 * x ** 2}

    return np.array(func[hp_type](delta_theta))


def resample_hourly_time_series(series: np.array, resolution):
    func = {96: lambda t: np.repeat(t, 4) if len(t) < 96 else t,
            1440: lambda t: np.repeat(t, 60) if len(t) < 1440 else t,
            24: lambda t: t if len(t) < 24 else t}
    return func[resolution](series)


class StandardLoadProfile:

    def __init__(self, demandQ: float, resolution: int = 96, consumer_type: str = 'EFH',
                 hp_type: str = 'ASHP', heating_system: str = 'radiator'):

        self.demandQ = demandQ
        self.consumer_type = consumer_type
        self.hp_type = hp_type
        self.heating_system = heating_system

        param = BUILDING_PARAMETERS[self.consumer_type]
        self.building_parameters = (param['A'], param['B'], param['C'], param['D'])
        self.mh = param['mh']
        self.bh = param['bh']
        self.mw = param['mw']
        self.bw = param['bw']

        self.day_values = REFERENCE_TEMPERATURE.groupby(REFERENCE_TEMPERATURE.index.day_of_year).mean().values.flatten()

        self.resolution = resolution  # -> set resolution

        self.A, self.B, self.C, self.D = self.building_parameters

        self.kw = self.demandQ / sum(self.get_h_value())

        self.temperature = deque([-15, 15, -15], maxlen=3)

        max_h_values = self.get_h_value(-20)
        max_factors = get_hourly_factors(mean_temp=-20, consumer_type=self.consumer_type)

        def round_to_base(value, base=5):
            return base * round(value / base) + base

        self.max_demand = round_to_base(max_h_values * self.kw * max(max_factors))

    def get_h_value(self, mean_temperature: float = None):
        if mean_temperature is None:
            sig = self.A / (1 + (self.B / (self.day_values - 40)) ** self.C) + self.D
            lin_h = self.mh * self.day_values + self.bh
            lin_w = self.mw * self.day_values + self.bw
            lin = np.max(np.vstack([lin_h, lin_w]), axis=0)
            return sig + lin
        else:
            sig = self.A / (1 + (self.B / (mean_temperature - 40)) ** self.C) + self.D
            lin_h = self.mh * mean_temperature + self.bh
            lin_w = self.mw * mean_temperature + self.bw
            lin = max(lin_h, lin_w)
            return sig + lin

    def get_water_heat_demand(self, mean_temperature: float = None):
        if mean_temperature > 15:
            return self.D + self.mw * mean_temperature + self.bw
        else:
            return self.D + self.mw * 15 + self.bw

    def run_model(self, d_time: pd.Timestamp, temperature: np.array = None) -> dict:

        r = {}

        if temperature is None:
            temperature = 5 * np.ones(24)

        hourly_temperature = copy(temperature)

        mean_temperature = float(np.mean(temperature))
        t1, t2, t3 = self.temperature

        temperature = (mean_temperature + 0.5 * t1 + 0.25 * t2 + 0.125 * t3) / (1 + 0.5 + 0.25 + 0.125)
        self.temperature.appendleft(temperature)

        hourly_factors = get_hourly_factors(temperature, day=d_time.day_name()) * \
                         DAILY_FACTORS[self.consumer_type][d_time.day_name()]

        total_heat_demand_at_day = self.kw * self.get_h_value(temperature)

        water_heat_demand_at_day = self.kw * self.get_water_heat_demand(temperature)
        hourly_water_heat_demand = hourly_factors * water_heat_demand_at_day
        r['hot_water'] = resample_hourly_time_series(hourly_water_heat_demand, self.resolution)
        dtheta = SINK_TEMPERATURE['hot_water'](hourly_temperature) - hourly_temperature
        r['cop_hot_water'] = resample_hourly_time_series(get_cop_time_series(dtheta, self.hp_type), self.resolution)

        space_heat_demand_at_day = total_heat_demand_at_day - water_heat_demand_at_day
        hourly_space_heat_demand = hourly_factors * space_heat_demand_at_day
        r['space_heating'] = resample_hourly_time_series(hourly_space_heat_demand, self.resolution)
        dtheta = SINK_TEMPERATURE[self.heating_system](hourly_temperature) - hourly_temperature
        r['cop_space_heating'] = resample_hourly_time_series(get_cop_time_series(dtheta, self.hp_type), self.resolution)

        return r


if __name__ == '__main__':
    import matplotlib.pyplot as plt




    weather = pd.read_csv(r'weather.csv', index_col=0, parse_dates=True)
    heatGen = StandardLoadProfile(demandQ=3748141, consumer_type='MFH', resolution=24)
    heat_space, hot_water = 0, 0

    fig, ax1 = plt.subplots()

    ax2 = ax1.twinx()

    for day in np.unique(weather.index.date)[1:]:
        temperature = weather.loc[weather.index.date == day, 'temp_air'].values
        r_dict = heatGen.run_model(pd.to_datetime(day), temperature)
        heat_space += r_dict['space_heating'].sum()
        hot_water += r_dict['hot_water'].sum()

        ts.append(r_dict['space_heating'])
        temp.append(temperature)

    ts = np.array(ts).flatten()
    temp = np.array(temp).flatten()

    ax1.plot(ts, 'b-')
    ax2.plot(temp, 'r-')
    t1 = np.unique(weather.index.date)[1]
    result = pd.DataFrame(data={'demand': ts, 'temp': temp},
                          index=pd.date_range(start=t1, freq='h', periods=len(ts)))
    result.to_excel('MFH.xlsx')

    #plt.plot(ts)
    #plt.plot(temp)

    plt.show()