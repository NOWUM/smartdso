import numpy as np
import pandas as pd

REFERENCE_TEMPERATURE = np.load(r'./demLib/data/heat_demand/reference_temperature.bin')
REFERENCE_TEMPERATURE = pd.DataFrame(data=dict(temp_air=REFERENCE_TEMPERATURE),
                                     index=pd.date_range(start='2018-01-01', freq='h',
                                                         periods=len(REFERENCE_TEMPERATURE)))
HOURLY_FACTORS = np.load(r'./demLib/data/heat_demand/hourly_factors.bin')


def get_hourly_factors(mean_temp: float):
    if mean_temp > 0:
        column = min(int(mean_temp / 5) + 4, 9)
    else:
        column = max(int(mean_temp / 5) + 3, 0)

    return HOURLY_FACTORS[:, column]


class StandardLoadProfile:

    def __init__(self, demandQ: float, building_parameters: tuple = (2.8, -37.5, 5.4, 0.17), resolution: int = 96):
        self.demandQ = demandQ
        self.building_parameters = building_parameters
        self.day_values = REFERENCE_TEMPERATURE.groupby(REFERENCE_TEMPERATURE.index.day_of_year).mean().values.flatten()

        self.resolution = resolution  # -> set resolution

        self.A, self.B, self.C, self.D = self.building_parameters

        self.kw = self.demandQ/sum(self.get_h_value())

    def get_h_value(self, mean_temperature: float = None):
        if mean_temperature is None:
            return self.A / (1 + (self.B / (self.day_values - 40)) ** self.C) + self.D
        else:
            return self.A / (1 + (self.B / (mean_temperature - 40)) ** self.C) + self.D

    def run_model(self, temperature: np.array = None):
        if temperature is None:
            temperature = 5*np.ones(24)
        mean_temperature = float(np.mean(temperature))
        heat_demand_at_day = self.kw * self.get_h_value(mean_temperature)

        hourly_heat_demand = get_hourly_factors(mean_temperature) * heat_demand_at_day

        if self.resolution == 96:
            return np.repeat(hourly_heat_demand, 4)
        elif self.resolution == 1440:
            return np.repeat(hourly_heat_demand, 60)
        else:
            return hourly_heat_demand


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    heatGen = StandardLoadProfile(demandQ=40000)

    demand = heatGen.run_model()

    plt.plot(demand)
    plt.show()