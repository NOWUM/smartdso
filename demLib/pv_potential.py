from shapely import wkt
from pyproj import Geod
import geopandas as gpd
import overpass as osm
import pandas as pd
from collections import defaultdict
import numpy as np

# api = overpy.Overpass()
url = "https://maps.mail.ru/osm/tools/overpass/api/interpreter"
api = osm.API(endpoint=url, timeout=300, debug=False)
geo_d = Geod(ellps="WGS84")

df = pd.read_csv(r'./gridLib/data/grid_allocations.csv', index_col=0)
df['photovoltaic_potential'] = 0
total_nodes = pd.read_csv(r'./gridLib/data/export/nodes.csv', index_col=0)
consumer_nodes = df.loc[df['profile']=='H0','bus0']
nodes = total_nodes.loc[consumer_nodes]
coords = nodes.loc[:,['lon', 'lat']]
coords = coords.drop_duplicates()

areas = defaultdict(list)
counter = 0
for index, coord in coords.iterrows():
    query = f"way(around:{2}, {coord['lat']}, {coord['lon']})[building=yes];out qt geom;"
    geojson = api.get(query)
    if len(geojson['features']) > 0:
        id_ = geojson['features'][0]['id']
        data = gpd.GeoDataFrame.from_features(geojson)
        area = abs(geo_d.geometry_area_perimeter(wkt.loads(str(data.loc[0, 'geometry'].convex_hull)))[0])
    else:
        id_ = counter
        area = 50

    areas[id_].append([index, area])
    counter += 1

total_A = 0
for key, values in areas.items():
    A = values[0][1]
    total_A += A
    l_, index = [], []
    for c in values:
        for i, d in df.loc[(df['bus0'] == c[0]) & (df['profile'] == 'H0'), 'jeb'].items():
            l_ += [d]
            index += [i]
    pv_potential = list(A/sum(l_) * np.asarray(l_)/7)
    for i, pv in zip(index, pv_potential):
        df.at[i, 'photovoltaic_potential'] = pv

df.to_csv(r'./gridLib/data/grid_allocations.csv')


