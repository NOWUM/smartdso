import numpy as np
from sqlalchemy import create_engine
from pvlib.location import Location
import pandas as pd
import calendar


DATABASE_URI = 'postgresql://opendata:opendata@10.13.10.41:5432/weather'


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


class WeatherGenerator:

    def __init__(self):

        weather_database_uri = 'postgresql://opendata:opendata@10.13.10.41:5432/weather'
        self.weather = WeatherInterface('grid_simulation', weather_database_uri)
        self.location = Location(longitude=6.4794, latitude=50.8037)
        self.sun_position = self.location.get_solarposition(pd.date_range(start='1972-01-01 00:00', periods=8684,
                                                            freq='h'))

    def get_weather(self, date: pd.Timestamp, area: str):
        azimuth = self.sun_position.loc[self.sun_position.index.day_of_year == date.day_of_year, 'azimuth'].to_numpy()
        zenith = self.sun_position.loc[self.sun_position.index.day_of_year == date.day_of_year, 'zenith'].to_numpy()
        year_ = date.year
        if calendar.isleap(date.year):
            date = date.replace(year=2016)
        else:
            date = date.replace(year=2015)
        temp_air = self.weather.get_temperature_in_area(area, date)
        wind_speed = self.weather.get_wind_in_area(area, date)
        dhi = self.weather.get_direct_radiation_in_area(area, date)
        dni = self.weather.get_diffuse_radiation_in_area(area, date)
        df = pd.concat([temp_air, wind_speed, dhi, dni], axis=1)
        df['ghi'] = df['dhi'] + df['dni']
        df['azimuth'] = azimuth
        df['zenith'] = zenith
        df.index = map(lambda x: x.replace(year=year_), df.index)

        return df
