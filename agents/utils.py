from pvlib.location import Location
import pandas as pd

from interfaces.weather import WeatherInterface


class WeatherGenerator:

    def __init__(self):

        weather_database_uri = 'postgresql://opendata:opendata@10.13.10.41:5432/weather'
        self.weather = WeatherInterface('grid_simulation', weather_database_uri)
        self.location = Location(longitude=6.4794, latitude=50.8037)
        self.sun_position = self.location.get_solarposition(pd.date_range(start='1972-01-01 00:00', periods=8684,
                                                            freq='h'))

    def get_weather(self, date, area):
        azimuth = self.sun_position.loc[self.sun_position.index.day_of_year == date.day_of_year, 'azimuth'].to_numpy()
        zenith = self.sun_position.loc[self.sun_position.index.day_of_year == date.day_of_year, 'zenith'].to_numpy()

        temp_air = self.weather.get_temperature_in_area(area, date)
        wind_speed = self.weather.get_wind_in_area(area, date)
        dhi = self.weather.get_direct_radiation_in_area(area, date)
        dni = self.weather.get_diffuse_radiation_in_area(area, date)
        df = pd.concat([temp_air, wind_speed, dhi, dni], axis=1)
        df['ghi'] = df['dhi'] + df['dni']
        df['azimuth'] = azimuth
        df['zenith'] = zenith

        return df


if __name__ == "__main__":
    from sqlalchemy import create_engine

    database = 'postgresql://opendata:opendata@10.13.10.41:5432/weather'
    engine = create_engine(database, connect_args={"application_name": 'x'})

    query = "Select \"time\", \"wind_meridional\", \"wind_zonal\" from cosmo where nuts = 'DEA26' and time >= '2012-01-01 00:00:00'"

    df = pd.read_sql(query, engine)