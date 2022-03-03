import numpy as np
from sqlalchemy import create_engine
import pandas as pd


class WeatherInterface:

    def __init__(self, name, database_url):
        self.engine = create_engine(database_url, connect_args={"application_name": name})

    def get_param(self, param, date):
        data_avg = []
        with self.engine.begin() as connection:
            for timestamp in pd.date_range(start=date, periods=24, freq='h'):
                query = f"SELECT avg({param}) FROM cosmo WHERE time = '{timestamp.isoformat()}';"
                res = connection.execute(query)
                value = res.fetchall()[0][0]
                data_avg.append({'time': timestamp, param: value})
        return pd.DataFrame(data_avg).set_index('time', drop=True)

    def get_param_in_area(self, param, area='DE111', date=pd.to_datetime('1995-1-1')):
        data_avg = []
        with self.engine.begin() as connection:
            for timestamp in pd.date_range(start=date, periods=24, freq='h'):
                query = f"SELECT {param} FROM cosmo WHERE time = '{timestamp.isoformat()}'" \
                        f"AND nuts = \'{area.upper()}\' ;"
                res = connection.execute(query)
                try:
                    value = res.fetchall()[0][0]
                except:
                    value = 0
                data_avg.append({'time': timestamp, param: value})
        return pd.DataFrame(data_avg).set_index('time', drop=True)

    def get_temperature_in_area(self, area='DE111', date=pd.to_datetime('1995-1-1')):
        return self.get_param_in_area('temp_air', area, date)

    def get_wind_in_area(self, area='DE111', date=pd.to_datetime('1995-1-1')):
        data_avg = []
        with self.engine.begin() as connection:
            for timestamp in pd.date_range(start=date, periods=24, freq='h'):
                query = f"SELECT wind_meridional, wind_zonal FROM cosmo " \
                        f"WHERE time = '{timestamp.isoformat()}'AND nuts = \'{area.upper()}\' ;"
                res = connection.execute(query)

                values = res.fetchall()[0]
                wind_speed = (values[0] ** 2 + values[1] ** 2) ** 0.5
                direction = np.arctan2(values[0] / wind_speed, values[1] / wind_speed) * 180 / np.pi
                direction += 180
                direction = 90 - direction
                data_avg.append({'time': timestamp, 'wind_speed': wind_speed, 'direction': direction})

        return pd.DataFrame(data_avg).set_index('time', drop=True)

    def get_direct_radiation_in_area(self, area='DE111', date=pd.to_datetime('1995-1-1')):
        return self.get_param_in_area('dhi', area, date)

    def get_diffuse_radiation_in_area(self, area='DE111', date=pd.to_datetime('1995-1-1')):
        return self.get_param_in_area('dni', area, date)

    def get_wind(self, date=pd.to_datetime('1995-1-1')):
        data_avg = []
        with self.engine.begin() as connection:
            for timestamp in pd.date_range(start=date, periods=24, freq='h'):
                query = f"SELECT avg(wind_meridional), avg(wind_zonal) FROM cosmo " \
                        f"WHERE time = '{timestamp.isoformat()}';"
                res = connection.execute(query)

                values = res.fetchall()[0]
                wind_speed = (values[0] ** 2 + values[1] ** 2) ** 0.5
                direction = np.arctan2(values[0] / wind_speed, values[1] / wind_speed) * 180 / np.pi
                direction += 180
                direction = 90 - direction
                data_avg.append({'time': timestamp, 'wind_speed': wind_speed, 'direction': direction})

        return pd.DataFrame(data_avg).set_index('time', drop=True)

    def get_direct_radiation(self, date=pd.to_datetime('1995-1-1')):
        return self.get_param('dhi', date)

    def get_diffuse_radiation(self, date=pd.to_datetime('1995-1-1')):
        return self.get_param('dni', date)

    def get_temperature(self, date=pd.to_datetime('1995-1-1')):
        return self.get_param('temp_air', date)


if __name__ == "__main__":
    weather_database_uri = 'postgresql://opendata:opendata@10.13.10.41:5432/weather'
    interface_weather = WeatherInterface('test', weather_database_uri)

    mean_temp = interface_weather.get_temperature()
    wind = interface_weather.get_wind_in_area(area='DE111')
    radR = interface_weather.get_direct_radiation_in_area(area='DE111')
    radF = interface_weather.get_diffuse_radiation_in_area(area='DE111')
    x = pd.concat([mean_temp, wind], axis=1)
