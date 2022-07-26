import ezdxf
import sys
from matplotlib import pyplot as plt
import pandas as pd
import uuid
import logging
from gridLib.plotting import get_plot
from pyproj import Geod, Transformer
import shapely.wkt as converter

logger = logging.getLogger('dwg-converter')
# https://de.wikipedia.org/wiki/European_Petroleum_Survey_Group_Geodesy#:~:text=Kartenanbieter%20im%20Netz.-,Deutschland,f%C3%BCr%20Gau%C3%9F%2DKr%C3%BCger%20(4.
transformer = Transformer.from_crs("epsg:31466", "epsg:4326")

geod = Geod(ellps="WGS84")

line_parameters = dict(Niederspannug={'r': 0.255, 'x': 0.08},
                       Hausanschluss={'r': 0.225, 'x': 0.08})


def get_coord(x_, y_):
    return transformer.transform(y_, x_)


def read_file(path: str = r'./gridLib/data/import/alliander/Porselen_Daten_new.dxf'):
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
        shape = f'POINT ({x} {y})'
        if shape not in inserted.keys():
            id_ = uuid.uuid1()
            nodes_[id_] = {'v_nom': 0.4, 'voltage_id': '_', 'lon': x, 'lat': y, 'shape': shape, 'injection': False,
                           'type': 'consumer'}
            consumers_[consumer.uuid] = {'bus0': id_, 'lon': x, 'lat': y, 'shape': shape,
                                         'v_nom': 0.4}
            inserted[shape] = id_
        else:
            consumers_[consumer.uuid] = {'bus0': inserted[shape], 'lon': x, 'lat': y, 'shape': shape}

    for station in msp.query('INSERT[layer=="Stationen" | layer =="KVS"]').entities:
        x, y, _ = station.dxf.insert
        y, x = get_coord(x, y)
        shape = f'POINT ({x} {y})'
        id_ = uuid.uuid1()
        if station.dxf.layer == 'Stationen':
            nodes_[id_] = {'v_nom': 0.4, 'voltage_id': '_', 'lon': x, 'lat': y, 'shape': shape, 'injection': True,
                           'type': 'transformer'}
        else:
            nodes_[id_] = {'v_nom': 0.4, 'voltage_id': '_', 'lon': x, 'lat': y, 'shape': shape, 'injection': False,
                           'type': 'cds'}

    consumers_ = pd.DataFrame.from_dict(consumers_, orient='index')
    nodes_ = pd.DataFrame.from_dict(nodes_, orient='index')

    return consumers_, nodes_


def get_lines(data, nodes: pd.DataFrame):
    msp = data.modelspace()
    lines_ = {}
    for line in msp.query('POLYLINE[layer=="Niederspannug" | layer=="Hausanschluss"]').entities:
        lon_coords, lat_coords = [], []
        buses, new_ = [], {}
        line_string = 'LINESTRING '
        for i in [0, -1]:
            x, y, _ = line.vertices[i].dxf.location
            y, x = get_coord(x, y)
            shape = f'POINT ({x} {y})'
            if shape in nodes['shape'].values:
                bus = nodes.loc[nodes['shape'] == shape].index[0]
            else:
                bus = uuid.uuid1()
                new_[bus] = {'v_nom': 0.4, 'voltage_id': '_', 'injection': False,
                             'lon': x, 'lat': y, 'shape': shape, 'type': 'sleeve'}

            buses += [bus]

        for x, y, _ in line.points():
            y, x = get_coord(x, y)
            if '(' not in line_string:
                line_string += f'({x} {y},'
            else:
                line_string += f' {x} {y},'
            lon_coords += [x]
            lat_coords += [y]
        shape = line_string[:-1] + ')'

        technical_parameters = line_parameters[line.dxf.layer]
        line_len = geod.geometry_length(converter.loads(shape))

        nodes = pd.concat([nodes, pd.DataFrame.from_dict(new_, orient='index')], axis=0)

        lines_[line.uuid] = {'bus0': buses[0], 'bus1': buses[-1], 'lon_coords': lon_coords,
                             'lat_coords': lat_coords, 'v_nom': 0.4, 'shape': shape,
                             'length': line_len, 'r': line_len/1e3 * technical_parameters['r'],
                             'x': line_len/1e3 * technical_parameters['x']}

    return pd.DataFrame.from_dict(lines_, orient='index'), nodes


if __name__ == "__main__":
    d = read_file()
    for layer in d.layers:
        logger.warning(f'found layer: {layer.dxf.name}')
    consumers, n = get_consumers(d)
    lines, n = get_lines(d, n)

    for name, line in lines.iterrows():
        plt.plot(line['lon_coords'], line['lat_coords'], 'b')
    #plt.scatter(consumers['lon'], consumers['lat'])
    #plt.scatter(n.loc[n['type'] == 'sleeve', 'lon'], n.loc[n['type'] == 'sleeve', 'lat'])
    # plt.show()

    # lines.to_csv(r'./Gridlib/data/export/alliander/edges.csv')
    # n.to_csv('./Gridlib/data/export/alliander/nodes.csv')
    # consumers.to_csv('./Gridlib/data/export/alliander/consumers.csv')
    #
    fig = get_plot(nodes=n, consumers=consumers, edges=lines)
    fig.write_html(r'./Gridlib/data/export/alliander/grid.html')
