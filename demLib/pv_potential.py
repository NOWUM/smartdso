import geopandas as gpd
import overpass as osm
import pandas as pd
from collections import defaultdict
from shapely.geometry import Point
from pyproj import Transformer

OSM_URL = f'https://maps.mail.ru/osm/tools/overpass/api/interpreter'
OSM_API = osm.API(endpoint=OSM_URL, timeout=300, debug=False)

tr = Transformer.from_crs('EPSG:4326', 'EPSG:25832')

pv_potential = gpd.read_file(r'./demLib/data/pv_potential/dueren/Solarkataster-Potentiale_05358008_Dueren_EPSG25832_Shape.shp')

df = pd.read_csv(r'./gridLib/data/export/dem/consumers.csv', index_col=0)
df['pv'] = None
total_nodes = pd.read_csv(r'./gridLib/data/export/dem/nodes.csv', index_col=0)
consumer_nodes = df.loc[df['profile'] == 'H0', 'bus0']
nodes = total_nodes.loc[consumer_nodes]
coords = nodes.loc[:, ['lon', 'lat']]
coords = coords.drop_duplicates()

buildings = defaultdict(list)
counter = 0
for index, coord in coords.iterrows():
    query = f"way(around:{2}, {coord['lat']}, {coord['lon']})[building=yes];out qt geom;"
    geojson = OSM_API.get(query)
    orientations = []
    if len(geojson['features']) > 0:
        id_ = geojson['features'][0]['id']
        data = gpd.GeoDataFrame.from_features(geojson)
        x, y = data.loc[0, 'geometry'].centroid.xy
        x, y = x[0], y[0]
        lat, lon = tr.transform(y, x)   # -> first lat, second lon
        # print('searching building')
        building_id = None
        for i, d in pv_potential.iterrows():
            if Point(lat, lon).within(d['geometry']):
                building_id = d['geb_id']
                break
        if building_id is not None:
            for _, system in pv_potential.loc[pv_potential['geb_id'] == building_id].iterrows():
                buildings[index] += [dict(pdc0=system['kw_17'],
                                          surface_tilt=system['neigung'],
                                          surface_azimuth=system['richtung'])]

for id_, systems in buildings.items():
    consumers = df.loc[df['bus0'] == id_]
    total_demand = consumers['jeb'].sum()
    for index, consumer in consumers.iterrows():
        pv_systems = []
        for system in systems:
            power = system['pdc0'] * consumer['jeb'] / total_demand
            pv_systems += [dict(pdc0=round(power, 1),
                                surface_tilt=system['surface_tilt'],
                                surface_azimuth=system['surface_azimuth'] if system['surface_azimuth'] != -1 else 0)]
        df.at[index, 'pv'] = str(pv_systems)

df.to_csv(r'./gridLib/data/export/dem/consumers.csv')


