import logging
import secrets
import sys
import numpy as np

import ezdxf
import pandas as pd
import shapely.wkt as converter
from matplotlib import pyplot as plt
from pyproj import Geod, Transformer
from shapely import wkt
from shapely.geometry import LineString, Point
from shapely.ops import nearest_points, split
from gridLib.model import GridModel

from gridLib.plotting import get_plot

logger = logging.getLogger("smartdso.dwg-converter")
# https://de.wikipedia.org/wiki/European_Petroleum_Survey_Group_Geodesy#:~:text=Kartenanbieter%20im%20Netz.-,Deutschland,f%C3%BCr%20Gau%C3%9F%2DKr%C3%BCger%20(4.
transformer = Transformer.from_crs("epsg:31466", "epsg:4326")

geod = Geod(ellps="WGS84")

line_parameters = dict(
    Niederspannug={"r": 0.255, "x": 0.08, "s_nom": 0.420},
    Hausanschluss={"r": 0.225, "x": 0.08, "s_nom": 0.213},
)


def get_coord(x_, y_):
    return transformer.transform(y_, x_)


def read_file(path: str = r"./gridLib/data/import/alliander/Porselen_Daten_5.dxf"):
    try:
        doc = ezdxf.readfile(path)
    except IOError:
        print(f"Not a DXF file or a generic I/O error.")
        sys.exit(1)
    except ezdxf.DXFStructureError:
        print(f"Invalid or corrupted DXF file.")
        sys.exit(2)
    return doc


def get_consumers(data):
    msp = data.modelspace()
    consumers_ = {}
    nodes_ = {}
    inserted = {}
    for consumer in msp.query('INSERT[layer=="Hausanschluss"]').entities:
        # print(consumer.dxf.color)
        x, y, _ = consumer.dxf.insert.xy
        y, x = get_coord(x, y)
        shape = f"POINT ({x} {y})"
        if shape not in inserted.keys():
            id_ = secrets.token_urlsafe(8)
            nodes_[id_] = {
                "v_nom": 0.4,
                "voltage_id": "_",
                "lon": x,
                "lat": y,
                "shape": shape,
                "injection": False,
                "type": "consumer",
            }
            consumers_[consumer.uuid] = {
                "bus0": id_,
                "lon": x,
                "lat": y,
                "shape": shape,
                "v_nom": 0.4,
            }
            inserted[shape] = id_
        else:
            consumers_[consumer.uuid] = {
                "bus0": inserted[shape],
                "lon": x,
                "lat": y,
                "shape": shape,
            }

    for station in msp.query('INSERT[layer=="Stationen" | layer =="KVS"]').entities:
        x, y, _ = station.dxf.insert
        y, x = get_coord(x, y)
        shape = f"POINT ({x} {y})"
        id_ = secrets.token_urlsafe(8)
        if station.dxf.layer == "Stationen":
            nodes_[id_] = {
                "v_nom": 0.4,
                "voltage_id": "_",
                "lon": x,
                "lat": y,
                "shape": shape,
                "injection": True,
                "type": "transformer",
            }
        else:
            nodes_[id_] = {
                "v_nom": 0.4,
                "voltage_id": "_",
                "lon": x,
                "lat": y,
                "shape": shape,
                "injection": False,
                "type": "cds",
            }

    consumers_ = pd.DataFrame.from_dict(consumers_, orient="index")
    nodes_ = pd.DataFrame.from_dict(nodes_, orient="index")

    return consumers_, nodes_


def get_lines(data, nodes: pd.DataFrame):
    msp = data.modelspace()
    lines_ = {}
    for line in msp.query(
        'POLYLINE[layer=="Niederspannug" | layer=="Hausanschluss"]'
    ).entities:
        lon_coords, lat_coords = [], []
        buses, new_ = [], {}
        line_string = "LINESTRING "
        for i in [0, -1]:
            x, y, _ = line.vertices[i].dxf.location
            y, x = get_coord(x, y)
            shape = f"POINT ({x} {y})"
            if shape in nodes["shape"].values:
                bus = nodes.loc[nodes["shape"] == shape].index[0]
            else:
                bus = secrets.token_urlsafe(8)
                new_[bus] = {
                    "v_nom": 0.4,
                    "voltage_id": "_",
                    "injection": False,
                    "lon": x,
                    "lat": y,
                    "shape": shape,
                    "type": "sleeve",
                }

            buses += [bus]

        for x, y, _ in line.points():
            y, x = get_coord(x, y)
            if "(" not in line_string:
                line_string += f"({x} {y},"
            else:
                line_string += f" {x} {y},"
            lon_coords += [x]
            lat_coords += [y]
        shape = line_string[:-1] + ")"

        technical_parameters = line_parameters[line.dxf.layer]
        line_len = geod.geometry_length(converter.loads(shape))

        nodes = pd.concat([nodes, pd.DataFrame.from_dict(new_, orient="index")], axis=0)

        lines_[line.uuid] = {
            "bus0": buses[0],
            "bus1": buses[-1],
            "lon_coords": lon_coords,
            "lat_coords": lat_coords,
            "v_nom": 0.4,
            "shape": shape,
            "length": line_len,
            "r": line_len / 1e3 * technical_parameters["r"],
            "x": line_len / 1e3 * technical_parameters["x"],
            "s_nom": technical_parameters["s_nom"],
        }

    return pd.DataFrame.from_dict(lines_, orient="index"), nodes


def get_not_connected_nodes(nodes, lines):
    not_con = []
    for idx in nodes.index:
        if (idx not in lines["bus0"].values) and (idx not in lines["bus1"].values):
            not_con += [idx]
    return not_con


def split_lines_for_not_connected_nodes(idx, nodes, lines):

    drop_idx = []
    new_ = []

    for node_id in idx:
        lon, lat = nodes.loc[node_id, ["lon", "lat"]].values
        for index, line_data in lines.iterrows():
            if (lon in line_data.lon_coords) and (lat in line_data.lat_coords):
                line_geom = converter.loads(line_data["shape"])
                node_geom = converter.loads(nodes.loc[node_id, "shape"])
                new_lines = split(line_geom, node_geom)
                drop_idx += [index]
                # print(len(new_lines.geoms))
                for geom in new_lines.geoms:
                    lon_coords, lat_coords = geom.coords.xy
                    lon_start, lon_end = lon_coords[0], lon_coords[-1]
                    lat_start, lat_end = lat_coords[0], lat_coords[-1]
                    line_length = geod.geometry_length(geom) / 1e3
                    rel = line_length / line_data.length
                    bus0 = nodes.loc[
                        (n["lon"] == lon_start) & (n["lat"] == lat_start)
                    ].index[0]
                    bus1 = nodes.loc[
                        (n["lon"] == lon_end) & (n["lat"] == lat_end)
                    ].index[0]
                    data = dict(
                        bus0=bus0,
                        bus1=bus1,
                        v_nom=line_data.v_nom,
                        shape=str(geom),
                        length=line_length,
                        lon_coords=list(lon_coords),
                        lat_coords=list(lat_coords),
                        r=rel * line_data.r,
                        x=rel * line_data.x,
                        s_nom=line_data.s_nom,
                    )
                    new_ += [data]
    lines = lines.drop(index=drop_idx)
    new_lines = pd.DataFrame(new_)
    new_lines.index = [secrets.token_urlsafe(8) for _ in range(len(new_lines))]
    lines = pd.concat([lines, new_lines])

    return lines


def insert_point_in_nearest_line(point: Point, line_set: pd.DataFrame):

    min_distance = 1e9
    idx = None

    for index, line_data in line_set.iterrows():
        geom = converter.loads(line_data["shape"])
        distance = geom.distance(point)
        if distance < min_distance:
            min_distance = distance
            idx = index

    lon_coords = lines.loc[idx, "lon_coords"]
    lat_coords = lines.loc[idx, "lat_coords"]

    min_distance = 1e9
    lon, lat = None, None

    nearest_line = converter.loads(line_set.loc[idx, "shape"])

    coord_idx = 0

    for coord in nearest_line.coords:
        distance = Point(coord).distance(point)
        if distance < min_distance:
            min_distance = distance
            coord_idx = lon_coords.index(coord[0])
            lon = coord[0]
            lat = coord[1]

    lon_coords = list(lon_coords)
    lon_coords.insert(coord_idx, lon)
    lat_coords = list(lat_coords)
    lat_coords.insert(coord_idx, lat)

    line_set.at[idx, "lon_coords"] = lon_coords
    line_set.at[idx, "lat_coords"] = lat_coords
    geom = LineString([(x, y) for x, y in zip(lon_coords, lat_coords)])
    line_set.at[idx, "shape"] = str(geom)

    return line_set


def get_transformers():

    tech_paras = {
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

    shapes = n.loc[n["type"] == "transformer", "shape"]
    index = n.loc[n["type"] == "transformer"].index
    len_ = len(n.loc[n["type"] == "transformer"].index)
    transformers = dict(
        bus0=index,
        v0=len_ * [0.4],
        bus1=[secrets.token_urlsafe(8) for _ in range(len_)],
        v1=len_ * [10],
        voltage_id=[secrets.token_urlsafe(8) for _ in range(len_)],
        lon=n.loc[n["type"] == "transformer", "lon"].values,
        lat=n.loc[n["type"] == "transformer", "lat"].values,
        shape=[shapes.loc[i] for i in index],
        r=[tech_paras[shape]["r"] for shape in [shapes.loc[i] for i in index]],
        x=[tech_paras[shape]["x"] for shape in [shapes.loc[i] for i in index]],
        g=[tech_paras[shape]["g"] for shape in [shapes.loc[i] for i in index]],
        b=[tech_paras[shape]["b"] for shape in [shapes.loc[i] for i in index]],
        s_nom=[tech_paras[shape]["s_nom"] for shape in [shapes.loc[i] for i in index]],
    )
    transformers = pd.DataFrame(transformers)

    return transformers


if __name__ == "__main__":
    d = read_file()
    for layer in d.layers:
        logger.warning(f"found layer: {layer.dxf.name}")
    consumers, n = get_consumers(d)
    lines, n = get_lines(d, n)

    not_connected = get_not_connected_nodes(n, lines)
    lines = split_lines_for_not_connected_nodes(not_connected, n, lines)
    not_connected = get_not_connected_nodes(n, lines)

    for node_id in not_connected:
        point = converter.loads(n.loc[node_id, "shape"])
        lines = insert_point_in_nearest_line(point, lines)

    lines = split_lines_for_not_connected_nodes(not_connected, n, lines)
    not_connected = get_not_connected_nodes(n, lines)

    index = (lines["bus0"].isin(consumers["bus0"].values)) | (
        lines["bus1"].isin(consumers["bus0"].values)
    )
    consumer_lines = lines.loc[index]
    not_connected_c = []
    double_consumers = []
    for index, data in consumer_lines.iterrows():
        if data.bus0 not in consumers["bus0"].values:
            if data.bus1 in consumers["bus0"].values:
                not_connected_c.append(data.bus0)
        elif data.bus1 not in consumers["bus0"].values:
            if data.bus0 in consumers["bus0"].values:
                not_connected_c.append(data.bus1)
        elif (
            data.bus0 in consumers["bus0"].values
            and data.bus1 in consumers["bus0"].values
        ):
            double_consumers.append(data)

    lines = split_lines_for_not_connected_nodes(not_connected_c, n, lines)

    not_connected_dc = []

    for line in double_consumers:
        found = False
        bus0 = converter.loads(n.loc[line.bus0, "shape"])
        bus1 = converter.loads(n.loc[line.bus1, "shape"])

        for index, connected_line in lines.iterrows():
            if line.name == index:
                break
            else:
                line_geom = converter.loads(connected_line["shape"])
                if line_geom.contains(bus0):
                    not_connected_dc.append(line.bus0)
                    found = True
                elif line_geom.contains(bus1):
                    not_connected_dc.append(line.bus1)
                    found = True
            if found:
                break

        if not found:
            min_distance = 1e9
            point = None
            for index, connected_line in lines.iterrows():
                if line.name == index:
                    break
                else:
                    line_geom = converter.loads(connected_line["shape"])

                    d1 = line_geom.distance(bus0)
                    d2 = line_geom.distance(bus1)

                    if (d1 < min_distance) or (d2 < min_distance):
                        distance = min(d1, d2)
                        line_index = index
                        point = bus0 if d2 > d1 else bus1

            if point == bus0:
                not_connected_dc.append(line.bus0)
            else:
                not_connected_dc.append(line.bus1)

            lines = insert_point_in_nearest_line(point, lines)

    lines = split_lines_for_not_connected_nodes(not_connected_dc, n, lines)

    transformers = get_transformers()

    # transformers.to_csv(r"./Gridlib/data/export/alliander/transformers.csv")

    model = GridModel(
        nodes=n,
        lines=lines,
        transformers=transformers,
        consumers=consumers
    )
    for id_ in np.unique(model.model.buses['sub_network'].values):
        print(len(model.model.buses.loc[model.model.buses['sub_network'] == id_, 'x'].values))
        plt.scatter(model.model.buses.loc[model.model.buses['sub_network'] == id_, 'x'].values,
                    model.model.buses.loc[model.model.buses['sub_network'] == id_, 'y'].values)
    plt.show()

    fig = get_plot(edges=lines, nodes=n, transformers=transformers, consumers=consumers)
    # n.to_csv("./Gridlib/data/export/alliander/nodes.csv")
    # consumers["profile"] = "H0"
    # consumers["jeb"] = 4500
    # consumers.to_csv("./Gridlib/data/export/alliander/consumers.csv")
    # lines["shape"] = lines["shape"].apply(converter.loads)
    # lines.to_csv(r"./Gridlib/data/export/alliander/lines.csv")
    #
    # fig = get_plot(nodes=n, edges=lines, transformers=transformers, consumers=consumers)
    # fig.write_html(r"./Gridlib/data/export/alliander/grid.html")
