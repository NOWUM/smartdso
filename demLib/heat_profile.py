import pandas as pd
import numpy as np
from numba import jitclass
from numba import int32, float32  # import the types

spec = [ ('temperature', float32[:]), ('factors', float32[:,:]), ('day_values', float32[:]), ('kw', float32),
        ('demandQ', float32), ('smooth',  int32), ('parameters', float32[:]), ('weights', float32[:]) ]

@jitclass(spec)
class slpGen:

    def __init__(self, demandQ=10000, parameters=np.asarray([2.8,-37,5.4,0.17], np.float32),
                 refTemp = np.asarray(np.load(open(r'data/Ref_Temp.array', 'rb')), np.float32),
                 factors = np.asarray(np.load(open(r'data/Ref_Factors.array', 'rb')), np.float32)):

        self.temperature = refTemp
        self.factors = factors

        self.demandQ = demandQ
        self.parameters = parameters
        self.smooth = 3

        self.day_values = np.asarray([np.mean(refTemp[i:i + 23]) for i in range(0, 8760, 24)],np.float32)

        h = self.parameters[0] / (1 + ((self.parameters[1] / (self.day_values - 40))) ** self.parameters[2]) + \
            self.parameters[3]

        self.kw = self.demandQ/np.sum(h)

    def get_profile(self, data=10.2*np.ones(24)):

        day_temp = np.mean(data)

        h = self.parameters[0] / (1 + ((self.parameters[1] / (day_temp - 40))) ** self.parameters[2]) + self.parameters[3]

        day_heat = float(h * self.kw)

        if day_temp <= -15:
            factors = self.factors[:, 0]
        if (day_temp > -15) & (day_temp <= -10):
            factors = self.factors[:, 1]
        if (day_temp > -10) & (day_temp <= -5):
            factors = self.factors[:, 2]
        if (day_temp > -5) & (day_temp <= 0):
            factors = self.factors[:, 3]
        if (day_temp > 0) & (day_temp <= 5):
            factors = self.factors[:, 4]
        if (day_temp > 5) & (day_temp <= 10):
            factors = self.factors[:, 5]
        if (day_temp > 10) & (day_temp <= 15):
            factors = self.factors[:, 6]
        if (day_temp > 15) & (day_temp <= 20):
            factors = self.factors[:, 7]
        if (day_temp > 20) & (day_temp <= 25):
            factors = self.factors[:, 8]
        if day_temp > 25:
            factors = self.factors[:, 9]

        hours = [day_heat * factor for factor in factors]

        return hours

if __name__ == '__main__':

    num = 2018
    year = pd.date_range(start='01.05.' + str(num), end='31.12.' + str(num), freq='d')
    index = pd.date_range(start='%i-01-01' % num, end='%i-12-31 23:00:00' % num, freq='60min')
    Qmax=[]
    heatGen = slpGen(demandQ=40000)
    lst = [heatGen.get_profile() for date in year]
