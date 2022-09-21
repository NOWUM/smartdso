import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point, Polygon
from collections import defaultdict
from tqdm import tqdm
import plotly.graph_objs as go

API_KEY = 'pk.eyJ1Ijoicmlla2VjaCIsImEiOiJjazRiYTdndXkwYnN3M2xteGN2MHhtZjB0In0.33tSDK45TXF3lb3-G147jw'


HEAT_DEMAND = gpd.read_file(r'./demLib/data/heat_demand/heinsberg')
HEAT_DEMAND = HEAT_DEMAND[[x is not None for x in HEAT_DEMAND['GEBAEUDETY'].values]]
HEAT_DEMAND.set_crs('epsg:25832', inplace=True)
HEAT_DEMAND.to_crs('epsg:4326', inplace=True)
BUILDINGS = pd.read_csv(r'./demLib/data/heat_demand/buildings.csv', sep=';', index_col=0)


consumers = pd.read_csv(r'./gridLib/data/export/alliander/consumers.csv', index_col=0)
# consumer_nodes = consumers.loc[consumers['profile'] == 'H0']
coords = consumers.loc[:, ['lon', 'lat']]
coords = coords.drop_duplicates()

min_lat, max_lat = coords['lat'].values.min(), coords['lat'].values.max()
min_lon, max_lon = coords['lon'].values.min(), coords['lon'].values.max()

bounding_box = Polygon([[min_lon, max_lat], [max_lon, max_lat],
                        [max_lon, min_lat], [min_lon, min_lat]])

bounding_box = gpd.GeoSeries(bounding_box)
bounding_box.set_crs('epsg:4326', inplace=True)

HEAT_DEMAND = gpd.overlay(HEAT_DEMAND, gpd.GeoDataFrame(bounding_box, columns=['geometry']), how='intersection')
# residential_types = ['EFH/DHH', 'RH', 'MFH', 'GMFH']

demand = dict()

for consumer_id, coord in tqdm(coords.iterrows(), total=coords.shape[0]):
    lon, lat = coord.lon, coord.lat
    rows = HEAT_DEMAND.loc[HEAT_DEMAND['geometry'].apply(lambda x: x.contains(Point((lon, lat))))]
    if rows.empty:
        distance = HEAT_DEMAND['geometry'].apply(lambda x: x.distance(Point((lon, lat))))
        rows = HEAT_DEMAND.loc[distance == distance.min()]

    building_type = rows['GEBAEUDETY'].values[0]
    building_years = BUILDINGS.loc[building_type]
    heat_demand = rows['spez_WB_HU'].values[0]
    building_years = building_years.loc[building_years > heat_demand]
    if building_years.empty:
        year = 1918
    else:
        year = float(building_years.index[-1])
    # includes 13.5 kWh/m²a warm water demand
    demand[consumer_id] = (rows['WB_HU'].values[0], year)

consumers['demandQ'] = 0
consumers['year'] = 1918

for id_, values in demand.items():
    consumers.at[id_, 'demandQ'] = values[0]
    consumers.at[id_, 'year'] = values[1]


# lons, lats = [], []
# for lon, lat in not_found.values():
#     lons.append(lon)
#     lats.append(lat)
#
#
# fig = go.Figure()
# trace = go.Scattermapbox(name=f'not found consumers', mode='markers',
#                          lon=lons, lat=lats)
#
# fig.add_trace(trace)
#
# fig.update_layout(mapbox=dict(accesstoken=API_KEY, bearing=0, pitch=0, zoom=16, style='outdoors',
#                               center=go.layout.mapbox.Center(lat=np.mean(lats), lon=np.mean(lons))),
#                   autosize=True)
#
# fig.write_html('not_found.html')