from pvlib.location import Location
import pandas as pd
import numpy as np


class WeatherGenerator:

    def __init__(self, lon: float = 6.4794, lat: float = 50.8037, path_to_weather_file: str = r'./weather.csv'):
        self.location = Location(longitude=lon, latitude=lat)
        self.weather = pd.read_csv(path_to_weather_file, index_col=0, parse_dates=True)

    def get_weather(self, date: pd.Timestamp):
        solar_const = 1368
        ghi = self.weather.loc[self.weather.index.date == date.date(), 'ghi']
        # pressure = self.weather.loc[self.weather.index.date == date.date(), 'pressure']
        sun_position = self.location.get_solarposition(ghi.index)
        elevation = sun_position['elevation'].values
        kt = ghi/(solar_const*np.sin(elevation/180 * np.pi))
        kd = []
        for k, gamma in zip(list(kt), list(elevation)):
            if k <= 0.3:
                kd += [1.02-0.254 * k + 0.0123 * np.sin(gamma/180 * np.pi)]
            elif k < 0.78:
                kd += [1.4 - 1.749 * k + 0.177 * np.sin(gamma/180 * np.pi)]
            else:
                kd += [0.486 * k - 0.182 * np.sin(gamma/180 * np.pi)]

        dhi = np.array(kd) * ghi
        dni = (ghi.values - dhi)

        weather = dict(
            dni=dni,
            dhi=dhi,
            ghi=ghi.values,
            zenith=sun_position['zenith'].values,
            azimuth=sun_position['azimuth'].values,
            wind_speed=self.weather.loc[self.weather.index.date == date.date(), 'wind_speed'].values,
            temp_air=self.weather.loc[self.weather.index.date == date.date(), 'temp_air'].values
        )

        return pd.DataFrame.from_dict(weather)


if __name__ == '__main__':
    myGen = WeatherGenerator()
    df = myGen.get_weather(pd.to_datetime('2022-03-01'))
