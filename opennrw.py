from multiprocessing import Pool
import pandas as pd
import requests
import re
import zipfile
import io
import geopandas as gpd
from sqlalchemy import create_engine
from shapely.geometry.polygon import Polygon
from shapely.geometry.multipolygon import MultiPolygon
import os
import glob
from uuid import uuid1

DATABASE_URI = 'postgresql://opendata:opendata@10.13.10.41:5432/opennrw'
ENGINE = create_engine(DATABASE_URI)

SOLAR_URL = 'https://www.opengeodata.nrw.de/produkte/umwelt_klima/klima/solarkataster/photovoltaik/'
HEAT_URL = 'https://www.opengeodata.nrw.de/produkte/umwelt_klima/klima/raumwaermebedarfsmodell/'

BASE_URLS = dict(solar=SOLAR_URL, heat=HEAT_URL)

CATEGORY = 'solar'

NUTS = gpd.read_file(r'./NUTS_RG_10M_2021_4326.shp')
NUTS3_DE = NUTS[(NUTS['CNTR_CODE'] == 'DE') & (NUTS['LEVL_CODE'] == 3)]
# -> build Nuts dataframe
NUTS.columns = map(str.lower, NUTS.columns)
NUTS = NUTS.loc[:, ['nuts_id', 'levl_code', 'cntr_code', 'name_latn', 'geometry']]
NUTS.columns = ['nuts_id', 'level', 'country', 'name', 'geometry']
NUTS['geometry'] = [MultiPolygon([feature]) if isinstance(feature, Polygon)
                    else feature for feature in NUTS["geometry"]]
NUTS.to_crs('epsg:4326', inplace=True)

# -> create table
ENGINE.execute("CREATE EXTENSION IF NOT EXISTS postgis;")
ENGINE.execute("CREATE EXTENSION IF NOT EXISTS postgis_raster;")
ENGINE.execute("CREATE TABLE IF NOT EXISTS nuts( "
               "nuts_id text, "
               "level integer, "
               "country text, "
               "name text, "
               "geometry geometry(MultiPolygon, 4326), "
               "PRIMARY KEY (nuts_id));")


def create_table(category: str = 'solar'):
    ENGINE.execute("CREATE EXTENSION IF NOT EXISTS postgis;")
    ENGINE.execute("CREATE EXTENSION IF NOT EXISTS postgis_raster;")
    if category == 'solar':
        ENGINE.execute("CREATE TABLE IF NOT EXISTS solar( "
                       "azimuth double precision, "
                       "tilt double precision, "
                       "power double precision, "
                       "energy double precision, "
                       "nuts_id text, "
                       "geb_id text, "
                       "name text, "
                       "geometry geometry(MultiPolygon, 4326), "
                       "PRIMARY KEY (geb_id, nuts_id));")

    elif category == 'heat':
        ENGINE.execute("CREATE TABLE IF NOT EXISTS heat( "
                       "typ text, "
                       "function text, "
                       "spec_demand double precision, "
                       "area double precision, "
                       "nuts_id text, "
                       "geb_id text, "
                       "demand double precision, "
                       "geometry geometry(MultiPolygon, 4326), "
                       "PRIMARY KEY (geb_id, nuts_id));")


def get_all_shapes(category: str = 'solar'):
    urls = []
    if category == 'solar':
        response = requests.get(SOLAR_URL)
        html_response = response.content.decode('utf-8')
        urls = re.findall("Solarkataster-Potentiale-Photovoltaik.*_Shape.zip", html_response)
    elif category == 'heat':
        response = requests.get(HEAT_URL)
        html_response = response.content.decode('utf-8')
        urls = re.findall("Raumwaermebedarfsmodell-NRW.*_Shape.zip", html_response)

    return urls


def get_data(url: str, id_: str):
    try:
        shape_zip = requests.get(url)
        file = zipfile.ZipFile(io.BytesIO(shape_zip.content))
        file.extractall(f'tmp_{id_}.shape')
        df = gpd.read_file(f'tmp_{id_}.shape')
        df.set_crs('epsg:25832', inplace=True)
        df.to_crs('epsg:4326', inplace=True)
        file.close()
        return df
    except Exception as e:
        print(repr(e))
        print(f'can not download data from {url}')
        return pd.DataFrame()


def find_nuts_id(d: gpd.GeoDataFrame):
    for i, system in d.iterrows():
        for k, area in NUTS3_DE.iterrows():
            try:
                if system.geometry.within(area.geometry):
                    d.at[i, 'nuts_id'] = area.NUTS_ID
                    break
            except Exception as e:
                print(repr(e))
                print('nuts_id not found')
    return d


def delete_files(id_: str):
    for file in glob.glob(fr'./tmp_{id_}.shape/*'):
        os.remove(file)


def write_in_db(d: gpd.GeoDataFrame):
    try:
        d['geometry'] = [MultiPolygon([feature]) if isinstance(feature, Polygon)
                         else feature for feature in d["geometry"]]
        d.to_postgis('solar', ENGINE, if_exists='append')
    except Exception as e:
        print(repr(e))


def run(shape: str):
    id_ = shape.split('_')[1]
    data = get_data(url=BASE_URLS[CATEGORY] + shape, id_ = id_)
    if CATEGORY == 'solar':
        data = data.loc[:, ['richtung', 'neigung', 'kw_17', 'str_17', 'kreis_gn', 'geometry']]
        data.columns = ['azimuth', 'tilt', 'power', 'energy', 'name', 'geometry']
    elif CATEGORY == 'heat':
        data = data.loc[:, ['GEBAEUDETY', 'Kreisname', 'Funktion', 'spez_WB_HU', 'EBZ_Final',
                            'WB_HU', 'geometry']]
        data.columns = ['typ', 'name', 'function', 'spec_demand', 'area', 'demand', 'geometry']

    data['nuts_id'] = 'not_found'
    data = find_nuts_id(data)

    data['geb_id'] = [uuid1() for _ in range(len(data))]
    write_in_db(data)
    delete_files(id_=id_)


if __name__ == '__main__':

    # create_table(category=CATEGORY)

    # delete_files()

    shapes = get_all_shapes(category=CATEGORY)
    run(shapes[50])


# https://www.opengeodata.nrw.de/produkte/umwelt_klima/klima/solarkataster/photovoltaik/Solarkataster-Potentiale-Photovoltaik_05111000_Duesseldorf_EPSG25832_Shape.zip
# -->Select ST_ConcaveHull(ST_Collect(ST_Centroid(geometry)), 0.9) as geom, sum(power) from solar
# --> Select ST_ConcaveHull(ST_Union(ST_Centroid(geometry), 0.0125), 0.5) as geom, sum(power) as power, cast(sum(power) as varchar) from solar group by nuts_id
# --> Select ST_ConvexHull(geometry) from solar
# Select ST_ClusterDBSCAN(ST_Centroid(geometry), eps := 100, minpoints := 5) over() AS cid from solar