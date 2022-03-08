import numpy as np
from datetime import date as to_date
from datetime import datetime
import pandas as pd

class BasicParticipant:

    def __init__(self, T: int = 96, address: dict = None, grid_node: str = None, *args, **kwargs):
        """
        Each Participant is characterized by the following parameters:
            - a current date as datetime date (yyyy-mm-dd)
            - information about the time resolution as number of steps per day, step length and a array
              containing all steps
            - a input dictionary with weather data for the current day
            - a inout dictionary with price data for the current day
            - a output dictionary for generation and demand and the resulting cash_flow
            - a output time series for the summarized power
            - a output time series for the volume if a storage is included
            - a address dictionary to locate the system

        :param T:
            Step per day as integer default = 96 --> 15 min resolution
        """

        # time resolution information
        if address is None:
            address = dict(lat=None, lon=None, city=None, street=None, number=None)
        self.T, self.t, self.dt = T, np.arange(T), 1/60
        # input parameters and time series for the optimization
        self.weather, self.prices = {}, {}
        # optimization output time series
        self.generation, self.demand = None, None
        self.power = None
        # initial with zeros
        self._initialize_data()
        
        self.address = address
        self.grid_node = grid_node
        
    def _initialize_data(self):
        """
        reset/initialize the output dictionaries and time series for the next day

        :return:
        """
        self.generation = dict(total=np.zeros((self.T,), float))
        self.demand = dict(power=np.zeros((self.T,), float), heat=np.zeros((self.T,), float),
                           charged=np.zeros(self.T))
        self.power = np.zeros(self.T, float)

    def set_parameter(self, weather: dict = None, prices: dict = None):
        """
        :param weather:
        :param prices:
        :return:
        """
        self.weather = weather
        self.prices = prices

    def optimize(self, d_time: datetime):
        pass