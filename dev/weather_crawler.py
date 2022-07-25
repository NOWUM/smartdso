import pygrib
import geopandas as gpd
import pandas as pd
import numpy as np
from collections import defaultdict
from matplotlib import pyplot as plt

SHAPE_PATH = r'../NUTS_RG_10M_2021_4326.shp'
NUTS = 'DEA26'


if __name__ == '__main__':

    weather = pygrib.open('weather_2022.grib')
    shapes = gpd.read_file(SHAPE_PATH)
    shape = shapes.loc[shapes['NUTS_ID'] == NUTS, 'geometry'].values[0]
    bounds = shape.bounds
    data_dict = defaultdict(list)

    # -> the day starts with hour=1 and ends with hour=0, so subtract 1 to convert to 0-23
    # Info here:
    # https://confluence.ecmwf.int/display/CKB/ERA5-Land%3A+data+documentation#ERA5Land:datadocumentation-accumulationsAccumulations
    for w in weather:
        v, _, _ = w.data(lat1=bounds[1], lat2=bounds[3], lon1=bounds[0], lon2=bounds[2])
        if w['validityTime']/100 == 0:
            offset = 23
        else:
            offset = w['validityTime']/100 - 1

        timestamp = w.validDate + pd.DateOffset(hours=offset)
        data_dict[w.name] += [(timestamp, v.mean())]

    weather.close()

    data_ = []
    for key, values in data_dict.items():
        data_ += [pd.DataFrame(columns=[key], index=[timestamp for timestamp, _ in values],
                               data=[value for _, value in values])]

    data = pd.concat(data_, axis=1)
    solar_rad = np.ediff1d(data.loc[:, 'Surface solar radiation downwards'].values) / 3600
    solar_rad[solar_rad < 0] = 0
    data.loc[:, 'Surface solar radiation downwards'] = [0] + list(solar_rad)
    data = data.loc[:, 'Surface solar radiation downwards']
    data.columns = ['ghi']
    data.to_csv('weather.csv')
    #plt.plot(solar_rad)
    #plt.show()

