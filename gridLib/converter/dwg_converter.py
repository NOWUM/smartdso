import logging
import secrets

import ezdxf
import pandas as pd
from pyproj import Geod, Transformer
from shapely.geometry import LineString, Point
from shapely.ops import split

from gridLib.plotting import get_plot as show_figure


logger = logging.getLogger("smartdso.dwg-converter")

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

    def __init__(self, path: str = r"./gridLib/data/import/alliander/Porselen_Daten_5.dxf"):
        self.doc = ezdxf.readfile(path)
        self.model_space = self.doc.modelspace()
        # -> output dictionary for dataframes
        self.components = {
            "nodes": {},
            "consumers": {},
            "lines": {},
            "transformers": {}
        }

        self.geo_distance = Geod(ellps="WGS84")
        self.change_crs = Transformer.from_crs("epsg:31466", "epsg:4326")

        self._node_coords: dict[str, str] = {}

    def get_coord_in_crs(self, x: float, y: float):
        lat, lon = self.change_crs.transform(y, x)
        return lon, lat

    def _add_new_node(self, geom: Point, node_type: str = 'consumer'):
        # -> add new node for type:...
        id_ = secrets.token_urlsafe(8)
        self._node_coords[str(geom)] = id_
        self.components['nodes'][id_] = {
            "v_nom": 0.4,
            "voltage_id": "_",
            "lon": geom.x,
            "lat": geom.y,
            "shape": geom,
            "injection": True if node_type == 'transformer' else False,
            "type": node_type,
        }
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

            self.components['consumers'][consumer.uuid] = {
                "bus0": id_,
                "lon": lon,
                "lat": lat,
                "shape": geom,
                "v_nom": 0.4,
            }

    def _build_transformers(self):
        query = 'INSERT[layer=="Stationen" | layer =="KVS"]'
        for station in self.model_space.query(query).entities:
            x, y, _ = station.dxf.insert.xy
            lon, lat = self.get_coord_in_crs(x, y)
            geom = Point((lon, lat))
            id_ = self._add_new_node(geom, node_type='cds')
            if station.dxf.layer == "Stationen":
                self.components['nodes'][id_]["injection"] = True
                self.components['nodes'][id_]["type"] = 'transformer'
                tr_id = secrets.token_urlsafe(8)
                self.components['transformers'][tr_id] = {
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
                }

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

            self.components['lines'][line.uuid] = {
                "bus0": buses[0],
                "bus1": buses[-1],
                "v_nom": 0.4,
                "shape": geom,
                "length": line_len,
                "r": line_len * technical_parameters["r"],
                "x": line_len * technical_parameters["x"],
                "s_nom": technical_parameters["s_nom"],
            }

    def _check_connections(self):
        lines = pd.DataFrame.from_dict(self.components['lines'], orient='index')
        missing_connections = []
        for node in self._node_coords.values():
            if (node not in lines["bus0"].values) and (node not in lines["bus1"].values):
                logger.info(f" -> Node: {node} is not connected >> converter will add a new connection")
                missing_connections.append(node)
        return missing_connections

    def _add_node_coords_to_lines(self, nodes: list):
        lines = []
        for node in nodes:
            geom = self.components['nodes'][node]['shape']
            min_distance, idx = 1e9, None
            for name, line in self.components['lines'].items():
                distance = line['shape'].distance(geom)
                if distance < min_distance:
                    min_distance = distance
                    idx = name

            nearest_line = self.components['lines'][idx]
            lines.append(idx)
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

            new_line = LineString([(x, y) for x, y in zip(coords['lon'], coords['lat'])])
            self.components['lines'][idx]['shape'] = new_line

        return lines

    def _add_nodes_to_lines(self, nodes: list, lines: list):
        for node, line in zip(nodes, lines):
            node_geom = self.components['nodes'][node]['shape']
            line_geom = self.components['lines'][line]['shape']
            total_length = self.geo_distance.geometry_length(line_geom) / 1e3

            new_lines = split(line_geom, node_geom)

            for geom in new_lines.geoms:
                id_ = secrets.token_urlsafe(8)
                lon_coords, lat_coords = geom.coords.xy
                lon_start, lon_end = lon_coords[0], lon_coords[-1]
                lat_start, lat_end = lat_coords[0], lat_coords[-1]
                line_length = self.geo_distance.geometry_length(geom) / 1e3
                factor = line_length/total_length
                bus_0 = Point((lon_start, lat_start))
                bus_1 = Point((lon_end, lat_end))

                self.components['lines'][id_] = {
                    "bus0": self._node_coords[str(bus_0)],
                    "bus1": self._node_coords[str(bus_1)],
                    "v_nom": 0.4,
                    "shape": geom,
                    "length": line_length,
                    "r": factor * self.components['lines'][line]["r"],
                    "x": factor * self.components['lines'][line]["x"],
                    "s_nom": self.components['lines'][line]["s_nom"],

                }

            self.components['lines'].pop(line)

    def convert(self):
        self._build_consumers()
        self._build_transformers()
        self._build_lines()
        missing_connections = self._check_connections()
        lines = self._add_node_coords_to_lines(missing_connections)
        self._add_nodes_to_lines(missing_connections, lines)

        for key, values in self.components.items():
            self.components[key] = pd.DataFrame.from_dict(values, orient='index')

    def save(self, path: str = r"./data/export/alliander/"):
        try:
            self.components["nodes"].to_csv(f"{path}nodes.csv")
            self.components["lines"].to_csv(f"{path}lines.csv")
            self.components["transformers"].to_csv(f"{path}transformers.csv")
            self.components["consumers"].to_csv(f"{path}consumers.csv")

        except Exception as e:
            self._logger.error("Error while saving the files")
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

    converter = DWGConverter()
    converter.convert()
