import numpy as np


class HeatStorage:
    # -> 30 bis 48 Liter (BDEW) hot water per day ~ 50 liter/person and day
    def __init__(self, volume: float, d_theta: int, steps: int = 96, resolution: str = '15min'):
        self.volume = (volume * 0.997 * 4.19 * d_theta) / 3600
        self.loss = 10.5 * volume / 3600
        self.V0 = self.volume

        self.T, self.t, self.dt = steps, np.arange(steps), steps/24

        self.resolution = resolution

        self._planned_usage = np.zeros(self.T)
        self._final_usage = np.zeros(self.T)

    def set_planned_usage(self, usage: np.array):
        self._planned_usage = usage

    def get_planned_usage(self):
        return self._planned_usage

    def set_final_usage(self, usage: np.array):
        self._final_usage = usage
        self.V0 = self._final_usage[-1]

    def get_final_usage(self):
        return self._final_usage