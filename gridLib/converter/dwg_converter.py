import logging
import secrets
import uuid
from collections import defaultdict

import ezdxf
import pandas as pd
from pyproj import Geod, Transformer
from shapely.geometry import LineString, Point
from shapely.ops import split

from gridLib.plotting import get_plot as show_figure

logging.basicConfig()
logger = logging.getLogger("smartdso.dwg-converter")
logger.setLevel('DEBUG')


TECHNICAL_SET_TRANSFORMERS = {
    "POINT (6.175021933947305 51.042055589425004)": dict(
        r=1.7007936507936505, x=5.704221240422533, b=0, g=0, s_nom=0.4
    ),
    "POINT (6.172851373373173 51.03625077909204)": dict(
        r=2.891, x=5.704221240422533, b=0, g=0, s_nom=0.315
    ),
    "POINT (6.1720412245041985 51.03911540501609)": dict(
        r=1.7007936507936505, x=5.704221240422533, b=0, g=0, s_nom=0.25
    ),
    "POINT (6.168528186689798 51.04178758359878)": dict(
        r=2.891, x=5.704221240422533, b=0, g=0, s_nom=0.4
    ),
    "POINT (6.174942910499844 51.03671131402324)": dict(
        r=2.89154513, x=5.704221240422533, b=0, g=0, s_nom=0.63
    ),
}

TECHNICAL_SET_LINES = {
    "Niederspannug": {"r": 0.255, "x": 0.08, "s_nom": 0.420},
    "Hausanschluss": {"r": 0.225, "x": 0.08, "s_nom": 0.213},
}

line_parameters = dict(

)


class DWGConverter:

    def __init__(self, path: str = r"./gridLib/data/import/alliander/Porselen_Daten_7.dxf"):
        self.doc = ezdxf.readfile(path)
        self.model_space = self.doc.modelspace()
        # -> output dictionary for dataframes
        self.components = {
            "nodes": pd.DataFrame(columns=["v_nom", "voltage_id", "lon", "lat",
                                           "shape","injection", "type"]),
            "consumers": pd.DataFrame(columns=["bus0", "lon", "lat", "shape", "v_nom"]),
            "lines": pd.DataFrame(columns=["bus0", "bus1", "v_nom", "length", "shape",
                                           "r", "x", "s_nom"]),
            "transformers": pd.DataFrame(columns=["bus0", "v_nom", "v0", "bus1", "v1", "voltage_id",
                                                  "lon", "lat", "shape", "r", "x", "g", "b", "s_nom"])

        }

        self.geo_distance = Geod(ellps="WGS84")
        self.change_crs = Transformer.from_crs("epsg:31466", "epsg:4326")

        self._node_coords: dict[str, str] = {}
        self._added_lines = []

    def get_coord_in_crs(self, x: float, y: float):
        lat, lon = self.change_crs.transform(y, x)
        return lon, lat

    def _add_new_node(self, geom: Point, node_type: str = 'consumer'):
        # -> add new node for type:...
        id_ = secrets.token_urlsafe(8)
        self._node_coords[str(geom)] = id_
        self.components['nodes'].at[id_] = pd.Series({
            "v_nom": 0.4,
            "voltage_id": "_",
            "lon": geom.x,
            "lat": geom.y,
            "shape": geom,
            "injection": True if node_type == 'transformer' else False,
            "type": node_type,
        })
        return id_

    def _build_consumers(self):
        query = "INSERT[layer==\"Hausanschluss\"]"
        for consumer in self.model_space.query(query).entities:
            x, y, _ = consumer.dxf.insert.xy
            lon, lat = self.get_coord_in_crs(x, y)
            geom = Point((lon, lat))
            if str(geom) in self._node_coords.keys():
                id_ = self._node_coords[str(geom)]
            else:
                id_ = self._add_new_node(geom)

            self.components['consumers'].at[consumer.uuid] = pd.Series({
                "bus0": id_,
                "lon": lon,
                "lat": lat,
                "shape": geom,
                "v_nom": 0.4,
            })

    def _build_transformers(self):
        query = 'INSERT[layer=="Stationen" | layer =="KVS"]'
        for station in self.model_space.query(query).entities:
            x, y, _ = station.dxf.insert.xy
            lon, lat = self.get_coord_in_crs(x, y)
            geom = Point((lon, lat))

            if str(geom) in self._node_coords.keys():
                id_ = self._node_coords[str(geom)]
            else:
                id_ = self._add_new_node(geom, node_type='cds')

            if station.dxf.layer == "Stationen":
                self.components['nodes'].loc[id_, "injection"] = True
                self.components['nodes'].loc[id_, "type"] = 'transformer'
                tr_id = secrets.token_urlsafe(8)
                self.components['transformers'].at[tr_id] = pd.Series({
                    'bus0': id_,
                    "v_nom": 0.4,
                    'v0': 0.4,
                    'bus1': None,
                    'v1': 10,
                    'voltage_id': '_',
                    'lon': geom.x,
                    'lat': geom.y,
                    'shape': geom,
                    'r': TECHNICAL_SET_TRANSFORMERS[str(geom)]['r'],
                    'x': TECHNICAL_SET_TRANSFORMERS[str(geom)]['x'],
                    'g': TECHNICAL_SET_TRANSFORMERS[str(geom)]['g'],
                    'b': TECHNICAL_SET_TRANSFORMERS[str(geom)]['b'],
                    's_nom': TECHNICAL_SET_TRANSFORMERS[str(geom)]['s_nom']
                })

    def _build_lines(self):
        query = 'POLYLINE[layer=="Niederspannug" | layer=="Hausanschluss"]'
        for line in self.model_space.query(query).entities:
            buses = {}
            for i in [0, -1]:
                # -> check starting point
                x, y, _ = line.vertices[i].dxf.location
                lon, lat = self.get_coord_in_crs(x, y)
                geom = Point((lon, lat))
                if str(geom) in self._node_coords.keys():
                    buses[i] = self._node_coords[str(geom)]
                else:
                    buses[i] = self._add_new_node(geom=geom, node_type='sleeve')

            coords = []
            for x, y, _ in line.points():
                lon, lat = self.get_coord_in_crs(x, y)
                coords.append((lon, lat))
            geom = LineString(coords)

            technical_parameters = TECHNICAL_SET_LINES[line.dxf.layer]
            line_len = self.geo_distance.geometry_length(geom) / 1e3

            self.components['lines'].at[line.uuid] = pd.Series({
                "bus0": buses[0],
                "bus1": buses[-1],
                "v_nom": 0.4,
                "shape": geom,
                "length": line_len,
                "r": line_len * technical_parameters["r"],
                "x": line_len * technical_parameters["x"],
                "s_nom": technical_parameters["s_nom"],
            })

    def _get_line_index(self, geom: Point, exclude_line=None):

        def lines_contain(g: LineString):
            lon_coords, lat_coords = g.coords.xy
            if geom.x in lon_coords and geom.y in lat_coords:
                return True
            return False

        df_lines = self.components['lines'].copy()
        if exclude_line is not None:
            df_lines = df_lines.drop([exclude_line])

        df_lines['contains'] = df_lines['shape'].apply(lines_contain)
        if any(df_lines['contains'].values):
            return df_lines.loc[df_lines['contains'].values].index[0], 'contains'
        else:
            df_lines['distance'] = df_lines['shape'].apply(lambda x: x.distance(geom))
            idx_min = df_lines['distance'] == df_lines['distance'].min()
            idx = df_lines.loc[idx_min].index.values[0]
            return idx, 'nearest'

    def _add_coord_to_line(self, idx: str, geom: Point):

        nearest_line = self.components['lines'].loc[idx]
        min_distance = 1e9
        coords = {'lon': [], 'lat': []}
        min_lon, min_lat = None, None

        for coord in nearest_line['shape'].coords:
            lon, lat = coord
            coords['lon'].append(lon)
            coords['lat'].append(lat)
            distance = Point(coord).distance(geom)
            if distance < min_distance:
                min_distance = distance
                min_lon, min_lat = lon, lat
        # -> add lon coord
        min_idx = coords['lon'].index(min_lon)
        coords['lon'].insert(min_idx, geom.x)
        # -> add lat coord
        min_idx = coords['lat'].index(min_lat)
        coords['lat'].insert(min_idx, geom.y)

        geom = LineString([(x, y) for x, y in zip(coords['lon'], coords['lat'])])
        self.components['lines'].at[idx, 'shape'] = geom

        return min_idx

    def _insert_node(self, idx, node_geom: Point, insert_coord: bool = False):

        if insert_coord:
            position = self._add_coord_to_line(idx, node_geom)
            line_geom = self.components['lines'].loc[idx, 'shape']
        else:
            line_geom = self.components['lines'].loc[idx, 'shape']
            position = list(line_geom.coords.xy[0]).index(node_geom.x)

        lon_coords, lat_coords = line_geom.coords.xy

        if position == 0:
            node_geom = Point((lon_coords[1], lat_coords[1]))
        elif position == len(line_geom.coords.xy[0]):
            node_geom = Point((lon_coords[-2], lat_coords[-2]))

        total_length = self.geo_distance.geometry_length(line_geom) / 1e3
        r = self.components['lines'].loc[idx, "r"]
        x = self.components['lines'].loc[idx, "x"]
        s_nom = self.components['lines'].loc[idx, "s_nom"]

        # -> build new lines
        geoms = split(line_geom, node_geom)

        for geom in geoms.geoms:

            id_ = secrets.token_urlsafe(8)
            lon_coords, lat_coords = geom.coords.xy
            lon_start, lon_end = lon_coords[0], lon_coords[-1]
            lat_start, lat_end = lat_coords[0], lat_coords[-1]
            line_length = self.geo_distance.geometry_length(geom) / 1e3
            bus_0 = Point((lon_start, lat_start))
            bus_1 = Point((lon_end, lat_end))

            factor = line_length / total_length

            self._added_lines.append(id_)

            self.components['lines'].at[id_] = pd.Series({
                "bus0": self._node_coords[str(bus_0)],
                "bus1": self._node_coords[str(bus_1)],
                "v_nom": 0.4,
                "shape": geom,
                "length": line_length,
                "r": factor * r,
                "x": factor * x,
                "s_nom": s_nom,
            })

    def _check_node_connections(self):
        lines = self.components['lines'].copy()
        for node in self._node_coords.values():
            if (node not in lines["bus0"].values) and (node not in lines["bus1"].values):
                logger.info(f" -> Node: {node} is not connected >> converter will add a new connection")
                node_geom = self.components['nodes'].loc[node, 'shape']
                # -> get nearest line
                idx, r = self._get_line_index(geom=node_geom)
                insert = True if r == 'nearest' else False
                self._insert_node(idx, node_geom, insert_coord=insert)
                self.components['lines'].drop([idx], inplace=True)

    def _check_line_connections(self):
        lines = self.components['lines']
        transformers = self.components['transformers']
        consumers = self.components['consumers']
        for index, line in lines.iterrows():
            tmp_lines = lines.copy()
            tmp_lines = tmp_lines.drop([index])
            connected_to = defaultdict(list)
            for bus in [0, 1]:
                node = line[f'bus{bus}']
                # node_type = self.components['nodes'][node]['type']
                if len(tmp_lines.loc[tmp_lines['bus0'] == node]) > 0:
                    connected_to[bus].append('l')
                if len(tmp_lines.loc[tmp_lines['bus1'] == node]) > 0:
                    connected_to[bus].append('l')
                if node in transformers['bus0'].values:
                    connected_to[bus].append('t')
                if node in consumers['bus0'].values:
                    connected_to[bus].append('c')
                if len(connected_to[bus]) == 0:
                    node_geom = self.components['nodes'].loc[node, 'shape']
                    # -> get nearest line
                    idx, r = self._get_line_index(node_geom, exclude_line=index)
                    insert = True if r == 'nearest' else False
                    self._insert_node(idx, node_geom, insert_coord=insert)
                    connected_to[bus].append('l')
                    return idx

            con_to_consumer_0 = connected_to[0] == ['c']
            con_to_consumer_1 = connected_to[1] == ['c']
            if con_to_consumer_0 and con_to_consumer_1:
                distances = {}
                for node in [line['bus0'], line['bus1']]:
                    node_geom = self.components['nodes'].loc[node,'shape']
                    idx, r = self._get_line_index(node_geom, exclude_line=index)
                    insert = True if r == 'nearest' else False
                    line_geom = self.components['lines'].loc[idx, 'shape']
                    distances[line_geom.distance(node_geom)] = (idx, node_geom, insert)
                self._insert_node(*distances[min(distances.keys())])
                return idx

        return 'checked'

    def convert(self):
        logger.info(" -> building consumers...")
        self._build_consumers()
        logger.info(" -> building transformers...")
        self._build_transformers()
        logger.info(" -> building lines...")
        self._build_lines()
        # -> check invalid nodes and fix them
        self._check_node_connections()
        # -> get invalid lines and fix them
        logger.info(' -> checking line connections...')
        status = ''
        while status != 'checked':
            status = self._check_line_connections()
            if status not in ['checked', 'warning']:
                self.components['lines'].drop([status], inplace=True)
                if status in self._added_lines:
                    self._added_lines.remove(status)

            logger.debug(f'status: {status}, number of lines: {len(self.components["lines"])}')

        # added_lines = self.components['lines'].loc[self._added_lines]
        # idx = added_lines.loc[added_lines['length'] > 0.150].index
        # self.components['lines'].drop(idx, inplace=True)

        for key, dataframe in self.components.items():
            dataframe.index.name = 'id_'
            self.components[key] = dataframe

    def save(self, path: str = r"./data/export/alliander/"):
        try:
            self.components["nodes"].to_csv(f"{path}nodes.csv")
            self.components["lines"].to_csv(f"{path}lines.csv")
            self.components["transformers"].to_csv(f"{path}transformers.csv")
            self.components["consumers"].to_csv(f"{path}consumers.csv")
            fig = self.plot()
            fig.write_html(f'{path}grid.html')
        except Exception as e:
            logger.error("Error while saving the files")
            print(repr(e))

    def plot(self):

        fig = show_figure(
            self.components["nodes"],
            self.components["lines"],
            self.components["transformers"],
            self.components["consumers"]
        )
        return fig


if __name__ == "__main__":
    from matplotlib import pyplot as plt
    from gridLib.model import GridModel
    import numpy as np

    converter = DWGConverter()
    converter.convert()

    converter.save(path=r'gridLib/data/export/alliander/')

    model = GridModel(
        nodes=converter.components['nodes'],
        lines=converter.components['lines'],
        transformers=converter.components['transformers'],
        consumers=converter.components['consumers']
    )

    for id_ in np.unique(model.model.buses['sub_network'].values):
        print(len(model.model.buses.loc[model.model.buses['sub_network'] == id_, 'x'].values))
        plt.scatter(model.model.buses.loc[model.model.buses['sub_network'] == id_, 'x'].values,
                    model.model.buses.loc[model.model.buses['sub_network'] == id_, 'y'].values)
    plt.show()
