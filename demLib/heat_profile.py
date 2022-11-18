import numpy as np
import pandas as pd
from collections import deque
from copy import copy


REFERENCE_TEMPERATURE = np.load(r"./demLib/data/heat_demand/reference_temperature.bin")
idx = pd.date_range(start="2018-01-01", freq="h", periods=len(REFERENCE_TEMPERATURE))
REFERENCE_TEMPERATURE = pd.DataFrame(data=dict(temp_air=REFERENCE_TEMPERATURE), index= idx)

HOURLY_FACTORS_EFH = pd.read_excel(r'./demLib/data/heat_demand/hourly_factors.xlsx', sheet_name='EFH')
HOURLY_FACTORS_EFH = HOURLY_FACTORS_EFH.fillna(method='ffill')
HOURLY_FACTORS_EFH = HOURLY_FACTORS_EFH.set_index(['Klasse', 'Niveau'])
HOURLY_FACTORS_MFH = pd.read_excel(r'./demLib/data/heat_demand/hourly_factors.xlsx', sheet_name='MFH')
HOURLY_FACTORS_MFH = HOURLY_FACTORS_MFH.fillna(method='ffill')
HOURLY_FACTORS_MFH = HOURLY_FACTORS_MFH.set_index(['Klasse', 'Niveau'])

SINK_TEMPERATURE = {'radiator': lambda x: 40 - 1.0 * x,
                    'floor': lambda x: 30 - 0.5 * x,
                    'hot_water': lambda x: np.array([50 for _ in x])}


def get_hourly_factors(mean_temp: float, class_: str = 'Klasse 4', resident_type: str = 'SFH'):
    if resident_type == 'SFH':
        factors = HOURLY_FACTORS_EFH
    else:
        factors = HOURLY_FACTORS_MFH

    factors = factors.loc[factors.index.get_level_values('Klasse') == class_]

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
    func = {96: lambda x: np.repeat(x, 4),
            1440: lambda x: np.repeat(x, 60),
            60: lambda x: x}
    return func[resolution](series)


class StandardLoadProfile:

    def __init__(self, demandQ: float, resolution: int = 96, resident_type: str = 'SFH',
                 hp_type: str = 'ASHP', heating_system: str = 'radiator'):

        self.demandQ = demandQ
        self.resident_type = resident_type
        self.hp_type = hp_type
        self.heating_system = heating_system

        if self.resident_type == 'SFH':
            # https://www.bdew.de/media/documents/Leitfaden_20160630_Abwicklung-Standardlastprofile-Gas.pdf
            # S. 133
            self.building_parameters = (1.6209544, -37.1833141, 5.6727847, 0.0716431)
            self.mh = -0.0495700
            self.bh = 0.8401015
            self.mw = -0.0022090
            self.bw = 0.1074468
        else:
            # https://www.bdew.de/media/documents/Leitfaden_20160630_Abwicklung-Standardlastprofile-Gas.pdf
            # S. 134
            self.building_parameters = (1.2328655, -34.7213605, 5.8164304, 0.0873352)
            self.mh = -0.0409284
            self.bh = 0.7672920
            self.mw = -0.0022320
            self.bw = 0.1199207

        self.day_values = REFERENCE_TEMPERATURE.groupby(REFERENCE_TEMPERATURE.index.day_of_year).mean().values.flatten()

        self.resolution = resolution  # -> set resolution

        self.A, self.B, self.C, self.D = self.building_parameters

        self.kw = self.demandQ / sum(self.get_h_value())

        self.temperature = deque([-15, 15, -15], maxlen=3)

        max_h_values = self.get_h_value(-20)
        max_factors = get_hourly_factors(mean_temp=-20, resident_type=self.resident_type)

        def round_to_base(value, base=5):
            return base * round(value / base)

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

    def run_model(self, temperature: np.array = None) -> dict:

        r = {}

        if temperature is None:
            temperature = 5 * np.ones(24)

        hourly_temperature = copy(temperature)

        mean_temperature = float(np.mean(temperature))
        t1, t2, t3 = self.temperature
        temperature = (mean_temperature + 0.5 * t1 + 0.25 * t2 + 0.125 * t3) / (1 + 0.5 + 0.25 + 0.125)
        self.temperature.appendleft(temperature)

        hourly_factors = get_hourly_factors(temperature)

        total_heat_demand_at_day = self.kw * self.get_h_value(temperature)

        water_heat_demand_at_day = self.kw * self.get_water_heat_demand(temperature)
        hourly_water_heat_demand = hourly_factors * water_heat_demand_at_day
        r['hot_water'] = resample_hourly_time_series(hourly_water_heat_demand, self.resolution)
        dtheta = SINK_TEMPERATURE['hot_water'](hourly_temperature) - hourly_temperature
        r['cop_hot_water'] = get_cop_time_series(dtheta, self.hp_type)

        space_heat_demand_at_day = total_heat_demand_at_day - water_heat_demand_at_day
        hourly_space_heat_demand = hourly_factors * space_heat_demand_at_day
        r['space_heating'] = resample_hourly_time_series(hourly_space_heat_demand, self.resolution)
        dtheta = SINK_TEMPERATURE[self.heating_system](hourly_temperature) - hourly_temperature
        r['cop_space_heating'] = get_cop_time_series(dtheta, self.hp_type)

        return r


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    weather = pd.read_csv(r'weather.csv', index_col=0, parse_dates=True)
    heatGen = StandardLoadProfile(demandQ=40000)
    x, y = 0, 0
    for temperature in heatGen.day_values:
        r_dict = heatGen.run_model(np.repeat(temperature, 24))
        x += r_dict['space_heating'].sum()
        y += r_dict['hot_water'].sum()

    print((x + y) / 4)
    #day_range = pd.date_range(start=weather.index[0], end=weather.index[-1], freq='d')
    #x, y = 0, 0
    #for day in day_range[1:]:
    #    r_dict = heatGen.run_model(temperature=weather.loc[weather.index.date == day, 'temp_air'].values)
    #    x += r_dict['space_heating'].sum()
    #    y += r_dict['hot_water'].sum()
    # demand = np.asarray(demand, dtype=float).flatten()
    # water_ = np.asarray(water_, dtype=float).flatten()
    # plt.plot(demand)
    # temperature = weather['temp_air'].groupby(weather.index.day_of_year).mean().values
    # plt.plot(temperature.repeat(96))
    # plt.show()
    #
    # water_ = np.asarray(water_, dtype=float).flatten()
    # plt.plot(water_)
    # temperature = weather['temp_air'].groupby(weather.index.day_of_year).mean().values
    # plt.plot(temperature.repeat(96))
    # plt.show()
