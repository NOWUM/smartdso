import numpy as np
from datetime import date as to_date
from datetime import datetime
import pandas as pd

class BasicParticipant:

    def __init__(self, T: int = 96, grid_node: str = None, *args, **kwargs):

        # time resolution information
        self.T, self.t, self.dt = T, np.arange(T), 1/(T/24)
        # input parameters and time series for the optimization
        self.weather, self.prices = {}, {}
        # optimization output time series
        self.generation, self.demand = None, None
        self.power = None
        # initial with zeros
        self._initialize_data()

        self.grid_node = grid_node
        
    def _initialize_data(self):

        self.generation = dict(total=np.zeros((self.T,), float))
        self.demand = dict(power=np.zeros((self.T,), float), heat=np.zeros((self.T,), float),
                           charged=np.zeros(self.T))
        self.power = np.zeros(self.T, float)

    def set_parameter(self, weather: dict = None, prices: dict = None):

        self.weather = weather
        self.prices = prices

    def optimize(self, d_time: datetime):
        pass