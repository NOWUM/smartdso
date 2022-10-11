import geopandas as gpd
import pandas as pd
from collections import defaultdict
from shapely.geometry import Point
from tqdm import tqdm

DATA_PATH = r'./demLib/data/pv_potential/dueren/Solarkataster-Potentiale_05358008_Dueren_EPSG25832_Shape.shp'

pv_potential = gpd.read_file(DATA_PATH)
pv_potential.set_crs('epsg:25832', inplace=True)
pv_potential.to_crs('epsg:4326', inplace=True)

consumers = pd.read_csv(r'./gridLib/data/export/dem/consumers.csv', index_col=0)
consumers['pv'] = None
consumers = consumers.loc[consumers['jeb'] > 0]
consumer_nodes = consumers.loc[consumers['profile'] == 'H0']
coords = consumer_nodes.loc[:, ['lon', 'lat']]
#coords = coords.drop_duplicates()

buildings = defaultdict(list)
for consumer_id, coord in tqdm(coords.iterrows(), total=coords.shape[0]):
    lon, lat = coord.lon, coord.lat
    
    #building = pv_potential.loc[pv_potential['geometry'].apply(lambda x: x.contains(Point((lon, lat))))]
    #if building.empty:
    pv_potential['distance'] = pv_potential['geometry'].apply(lambda x: x.distance(Point((lon, lat))))
    building = pv_potential.loc[pv_potential['distance'].min()==pv_potential['distance']]


    if not building.empty:
        for id_ in building['OBJECTID'].values:
            buildings[id_].append(consumer_id)


systems = defaultdict(list)
for building_id, consumer_ids in buildings.items():
    pv_system = pv_potential.loc[pv_potential['OBJECTID'] == building_id]
    power = pv_system['kw_17'].values[0]
    # peak power in kW with 17% wirkleistung
    surface_tilt = pv_system['neigung'].values[0]
    if pv_system['richtung'].values[0] != -1:
        surface_azimuth = pv_system['richtung'].values[0] 
    else:
        surface_azimuth = 180

    total_demand = consumers.loc[consumer_ids, 'jeb'].sum()
    # jeb = jahres energie bedarf

    for consumer_id in consumer_ids:
        pdc0 = (power/total_demand) * consumers.loc[consumer_ids, 'jeb'].values[0]
        systems[consumer_id].append(dict(pdc0=round(pdc0, 1), surface_tilt=surface_tilt,
                                         surface_azimuth=surface_azimuth))

for consumer_id, systems in systems.items():
    consumers.at[consumer_id, 'pv'] = str(systems)


consumers.to_csv(r'./gridLib/data/export/dem/consumers.csv')


# counter = 0
# for index, coord in coords.iterrows():
#     query = f"way(around:{2}, {coord['lat']}, {coord['lon']})[building=yes];out qt geom;"
#     geojson = OSM_API.get(query)
#     orientations = []
#     if len(geojson['features']) > 0:
#         id_ = geojson['features'][0]['id']
#         data = gpd.GeoDataFrame.from_features(geojson)
#         x, y = data.loc[0, 'geometry'].centroid.xy
#         x, y = x[0], y[0]
#         lat, lon = tr.transform(y, x)   # -> first lat, second lon
#         # print('searching building')
#         building_id = None
#         for i, d in pv_potential.iterrows():
#             if Point(lat, lon).within(d['geometry']):
#                 building_id = d['geb_id']
#                 break
#         if building_id is not None:
#             for _, system in pv_potential.loc[pv_potential['geb_id'] == building_id].iterrows():
#                 buildings[index] += [dict(pdc0=system['kw_17'],
#                                           surface_tilt=system['neigung'],
#                                           surface_azimuth=system['richtung'])]
#



