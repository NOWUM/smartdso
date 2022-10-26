from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
import pandas as pd
import os
from shapely.wkt import loads
import numpy as np
import secrets
from shapely.geometry import Point
from matplotlib import pyplot as plt


geolocator = Nominatim(user_agent=secrets.token_urlsafe(8))
geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1.2)


GRID_DATA = os.getenv('GRID_DATA', 'alliander')

data_path = fr'./gridLib/data/export/{GRID_DATA}'
total_consumers = pd.read_csv(fr'{data_path}/consumers.csv', index_col=0)


def convert_address(a: str):
    a_parts = a.split(' ')
    convert = True
    a = ''
    for part in a_parts:
        try:
            num = float(part)
            a += ' ' + part
            break
        except Exception as e:
            a += ' ' + part
    # a = a_parts[0] + ' ' + a_parts[1]
    adr_str = f'{a} 52525 Heinsberg'
    return adr_str


if __name__ == "__main__":

    df = pd.read_excel(r'./gridLib/data/import/alliander/comsumer_data.xlsx', sheet_name='Tabelle1')
    df = df.loc[df['KUGR_NAME'].isin(['H01', 'G01', 'G11', 'G21', 'G31', 'G41', 'G51', 'G61'])]
    df['ADRESSE'] = df['OBJEKTBEZEICHNUNG'].apply(convert_address)

    df['location'] = df['ADRESSE'].apply(geocode)
    df['lat'] = df['location'].apply(lambda x: x.latitude if x else None)
    df['lon'] = df['location'].apply(lambda x: x.longitude if x else None)
    df['shape'] = [Point(row.lon, row.lat) for _, row in df.iterrows()]

    maps = {}
    for consumer_id, consumer in total_consumers.iterrows():
        consumer_point = loads(consumer['shape'])
        distance = np.inf
        idx = None
        for index, row in df.iterrows():
            compare_point = row['shape']
            current_distance = consumer_point.distance(compare_point)
            if current_distance < distance:
                distance = current_distance
                idx = index
        maps[consumer_id] = idx

    total_consumers['jeb'] = 0

    for consumer_id,  index in maps.items():
        total_consumers.loc[consumer_id, 'jeb'] += df.loc[index, 'Jahresverbrauch']
