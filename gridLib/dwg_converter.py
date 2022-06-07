import ezdxf
import sys
from matplotlib import pyplot as plt
import pandas as pd
import uuid
from shapely import wkt
from shapely.geometry import Point
import logging

logger = logging.getLogger('dwg-converter')


def read_file(path: str = r'./gridLib/data/Porselen_Daten_2.dxf'):
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
    for consumer in msp.query('POINT[layer=="Hausanschluss"]').entities:
        x, y, _ = consumer.dxf.location.xy
        shape = f'POINT ({x}, {y})'
        if shape not in inserted.keys():
            id_ = uuid.uuid1()
            nodes_[id_] = {'v_nom': 0.4, 'voltage_id': '_', 'lon': x, 'lat': y, 'shape': shape}
            consumers_[consumer.uuid] = {'bus0': id_, 'lon': x, 'lat': y, 'shape': shape}
            inserted[shape] = id_
        else:
            consumers_[consumer.uuid] = {'bus0': inserted[shape], 'lon': x, 'lat': y, 'shape': shape}

    for station in msp.query('INSERT[layer=="Stationen"]').entities:
        x, y, _ = station.dxf.insert
        shape = f'POINT ({x}, {y})'
        id_ = uuid.uuid1()
        nodes_[id_] = {'v_nom': 0.4, 'voltage_id': '_', 'lon': x, 'lat': y, 'shape': shape}

    consumers_ = pd.DataFrame.from_dict(consumers_, orient='index')
    nodes_ = pd.DataFrame.from_dict(nodes_, orient='index')

    return consumers_, nodes_


def get_lines(data, nodes: pd.DataFrame):
    msp = data.modelspace()
    lines_ = {}
    not_found = {}
    lines = msp.query('POLYLINE[layer=="Niederspannug" | layer=="Hausanschluss"]')
    for line in lines.entities:
        lon_coords, lat_coords = [], []
        buses = []
        for i in [0, -1]:
            x, y, _ = line.vertices[i].dxf.location
            shape = f'POINT ({x}, {y})'
            if shape in nodes['shape'].values:
                bus = nodes.loc[nodes['shape'] == shape].index[0]
            else:
                bus = uuid.uuid1()
                node = pd.DataFrame.from_dict({'id': bus, 'v_nom': 0.4, 'voltage_id': '_',
                                               'lon': x, 'lat': y, 'shape': shape}, orient='index')
                nodes = pd.concat([nodes, node])
            buses += [bus]

        for x, y, _ in line.points():
            lon_coords += [x]
            lat_coords += [y]

        lines_[line.uuid] = {'bus0': buses[0], 'bus1': buses[-1], 'lon_coords': lon_coords,
                             'lat_coords': lat_coords}

    return pd.DataFrame.from_dict(lines_, orient='index'), nodes


if __name__ == "__main__":
    d = read_file()
    for layer in d.layers:
        logger.warning(f'found layer: {layer.dxf.name}')
    consumers, n = get_consumers(d)
    lines, n = get_lines(d, n)

    for name, line in lines.iterrows():
        plt.plot(line['lon_coords'], line['lat_coords'], 'b')
    plt.show()

    lines.to_csv(r'./Gridlib/data/export/alliander/edges.csv')
    n.to_csv('./Gridlib/data/export/alliander/nodes.csv')
    consumers.to_csv('./Gridlib/data/export/alliander/consumers.csv')
