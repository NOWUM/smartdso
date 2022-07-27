import pygrib
import geopandas as gpd
import pandas as pd
import numpy as np
from collections import defaultdict
from tqdm import tqdm
from sqlalchemy import create_engine

DATABASE_URI = 'postgresql://opendata:opendata@10.13.10.41:5432/weather'
ENGINE = create_engine(DATABASE_URI)

SHAPE_PATH = r'./NUTS_RG_10M_2021_4326.shp'

WEATHER_DATA = pygrib.open('./dev/weather_2022.grib')
SHAPES = gpd.read_file(SHAPE_PATH)


def create_table():
    ENGINE.execute("CREATE EXTENSION IF NOT EXISTS postgis;")
    ENGINE.execute("CREATE EXTENSION IF NOT EXISTS postgis_raster;")

    ENGINE.execute("CREATE TABLE IF NOT EXISTS ecmwf_eu( "
                   "time timestamp without time zone NOT NULL, "
                   "wind_speed double precision, "
                   "direction double precision, "
                   "ghi double precision, "
                   "pressure double precision, "
                   "temp_air double precision, "
                   "precipitation double precision, "
                   "nuts_id text, "
                   "geometry geometry(Polygon, 4326), "
                   "PRIMARY KEY (time , nuts_id));")

    query_create_hypertable = "SELECT create_hypertable('ecmwf_eu', 'time', if_not_exists => TRUE, migrate_data => TRUE);"
    with ENGINE.connect() as connection:
        with connection.begin():
            connection.execute(query_create_hypertable)


def get_data(nuts_id: str):
    shape = SHAPES.loc[SHAPES['NUTS_ID'] == nuts_id, 'geometry'].values[0]
    bounds = shape.bounds
    data_dict = defaultdict(list)

    for w in WEATHER_DATA:
        v, _, _ = w.data(lat1=bounds[1], lat2=bounds[3], lon1=bounds[0], lon2=bounds[2])
        if w['validityTime']/100 == 0:
            offset = 23
        else:
            offset = w['validityTime']/100 - 1

        timestamp = w.validDate + pd.DateOffset(hours=offset)
        data_dict[w.name] += [(timestamp, v.mean())]

    data_ = [pd.DataFrame(columns=[key], index=[timestamp for timestamp, _ in values],
                               data=[value for _, value in values]) for key, values in data_dict.items()]

    return pd.concat(data_, axis=1)


def build_export(data: pd.DataFrame):

    export_weather = pd.DataFrame(columns=['wind_speed', 'direction', 'ghi', 'pressure', 'temp_air', 'precipitation'],
                                  index=data.index)
    # -> get solar radiation
    solar_rad = np.ediff1d(data.loc[:, 'Surface solar radiation downwards'].values) / 3600
    solar_rad[solar_rad < 0] = 0
    export_weather.loc[:, 'ghi'] = [0] + list(solar_rad)

    # -> get temperature in °C
    export_weather.loc[:, 'temp_air'] = data.loc[:, '2 metre temperature'].values - 273.15

    # -> set pressure
    export_weather.loc[:, 'pressure'] = data.loc[:, 'Surface pressure'].values

    # -> get precipitation
    precipitation = np.ediff1d(data.loc[:, 'Total precipitation'].values)
    precipitation[precipitation < 0] = 0
    export_weather.loc[:, 'precipitation'] = [0] + list(precipitation)

    # -> get wind_speed and direction
    u_wind = data['10 metre U wind component'].values
    v_wind = data['10 metre V wind component'].values
    wind_speed = (u_wind**2 + v_wind**2)**0.5

    direction = np.arctan2(u_wind / wind_speed, v_wind / wind_speed) * 180 / np.pi
    direction += 180
    direction = 90 - direction

    export_weather.loc[:, 'wind_speed'] = wind_speed
    export_weather.loc[:, 'direction'] = direction

    return export_weather


if __name__ == '__main__':
    create_table()

    for n_id in tqdm(SHAPES['NUTS_ID'].unique()):
        print(n_id)
        weather_data = get_data(nuts_id=n_id)
        export = build_export(weather_data)

        shape = SHAPES.loc[SHAPES['NUTS_ID'] == n_id, 'geometry'].values[0]
        export['nuts_id'] = n_id
        export['geometry'] = shape
        export = gpd.GeoDataFrame(export)
        export = export.set_crs('epsg:4326')
        export.index.name = 'time'
        export = export.reset_index()
        export.to_postgis('ecmwf_eu', ENGINE, if_exists='append')
        break






