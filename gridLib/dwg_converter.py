import ezdxf
import sys
from matplotlib import pyplot as plt
import pandas as pd
import uuid
import logging
from gridLib.plotting import get_plot

logger = logging.getLogger('dwg-converter')


max_lon, min_lon = 6.1805, 6.1664
d_lon = max_lon-min_lon
max_lat, min_lat = 51.0441379485292, 51.03408
d_lat = max_lat-min_lat

max_x, min_x = 2512700.75396262, 2511715.72460938
dx = max_x-min_x
max_y, min_y = 5656576.519166669, 5655451.43618056
dy = max_y-min_y


def get_coord(x_, y_):
    x_ = (x_ - min_x) * d_lon/dx + min_lon
    y_ = (y_ - min_y) * d_lat/dy + min_lat
    return x_, y_


def read_file(path: str = r'./gridLib/data/import/alliander/Porselen_Daten_2.dxf'):
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
        x, y = get_coord(x, y)
        shape = f'POINT ({x} {y})'
        if shape not in inserted.keys():
            id_ = uuid.uuid1()
            nodes_[id_] = {'v_nom': 0.4, 'voltage_id': '_', 'lon': x, 'lat': y, 'shape': shape, 'injection': False,
                           'type': 'consumer'}
            consumers_[consumer.uuid] = {'bus0': id_, 'lon': x, 'lat': y, 'shape': shape,
                                         'v_nom': 0.4}
            inserted[shape] = id_
        else:
            consumers_[consumer.uuid] = {'bus0': inserted[shape], 'lon': x, 'lat': y, 'shape': shape,}

    for station in msp.query('INSERT[layer=="Stationen" | layer =="KVS"]').entities:
        x, y, _ = station.dxf.insert
        x, y = get_coord(x, y)
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
            x, y = get_coord(x, y)
            shape = f'POINT ({x} {y})'
            if shape in nodes['shape'].values:
                bus = nodes.loc[nodes['shape'] == shape].index[0]
            else:
                bus = uuid.uuid1()
                new_[bus] = {'v_nom': 0.4, 'voltage_id': '_', 'injection': False,
                             'lon': x, 'lat': y, 'shape': shape, 'type': 'sleeve'}

            buses += [bus]

        for x, y, _ in line.points():
            x, y = get_coord(x, y)
            if '(' not in line_string:
                line_string += f'({x} {y},'
            else:
                line_string += f' {x} {y},'
            lon_coords += [x]
            lat_coords += [y]
        shape = line_string[:-1] + ')'

        nodes = pd.concat([nodes, pd.DataFrame.from_dict(new_, orient='index')], axis=0)

        lines_[line.uuid] = {'bus0': buses[0], 'bus1': buses[-1], 'lon_coords': lon_coords,
                             'lat_coords': lat_coords, 'v_nom': 0.4, 'shape': shape}

    return pd.DataFrame.from_dict(lines_, orient='index'), nodes


if __name__ == "__main__":
    d = read_file()
    for layer in d.layers:
        logger.warning(f'found layer: {layer.dxf.name}')
    consumers, n = get_consumers(d)
    lines, n = get_lines(d, n)

    for name, line in lines.iterrows():
        plt.plot(line['lon_coords'], line['lat_coords'], 'b')
    plt.scatter(consumers['lon'], consumers['lat'])
    plt.scatter(n.loc[n['type'] == 'sleeve', 'lon'], n.loc[n['type'] == 'sleeve', 'lat'])
    # plt.show()

    lines.to_csv(r'./Gridlib/data/export/alliander/edges.csv')
    n.to_csv('./Gridlib/data/export/alliander/nodes.csv')
    consumers.to_csv('./Gridlib/data/export/alliander/consumers.csv')

    fig = get_plot(nodes=n, consumers=consumers, edges=lines)
    fig.write_html(r'./Gridlib/data/export/alliander/grid.html')
