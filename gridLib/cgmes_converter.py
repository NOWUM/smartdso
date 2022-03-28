import glob as gb
import cimpy
import logging
from collections import defaultdict
import numpy as np
import pandas as pd
from shapely.geometry import Point, LineString
import pickle

from gridLib.plotting import show_plot as show_figure
from gridLib.geo_information import GeoInformation



def component_type(component):
    return component.__class__.__name__


class CGMESConverter:

    def __init__(self, path: str = r'./data/import/', levels: tuple = (0.4, 10, 20)):
        """
        A simple cgmes to csv converter with output in  pypsa style:
            convert xml file, which are located in the directory path, to csv files.
            The converter creates four files to build up a pypsa network.
            - nodes.csv with columns: [id, v_nom, voltage_id, lon, lat, shape, injection]
            - edges.csv with columns: [id, bus0, bus1, s_nom, r, x, len,voltage_id, lon_coords, lat_coords, shape]
            - transformers.csv with columns [id, bus0, v0, bus1, v1, s_nom, r, x, b, g voltage_id, lat, lon, shape]
            - consumers.csv with columns []
        Parameters
        ----------
        :param path: str
            path to file directory
        :param levels: tuple
            voltage levels which are includes in files e.g. (0.4, 10, 20)
        """

        self._logger = logging.getLogger('converter')                   # converting logging
        self._working_path = path                                       # path where the xml file are stored
        self._sub_paths = {'0.4': fr'{self._working_path}400V/*.xml',
                           '10': fr'{self._working_path}10kV/*.xml',
                           '20': fr'{self._working_path}20kV/*.xml' }
        self._voltage_levels = levels                                   # maximal voltage level

        # load data from source
        def flatten(t):
            return [item for sublist in t for item in sublist]
        files_to_convert = flatten([gb.glob(self._sub_paths[str(level)]) for level in self._voltage_levels])
        import_result = cimpy.cim_import(files_to_convert, "cgmes_v2_4_15")
        self.grid_data = import_result['topology']                      # total grid data resulting from xml files

        self.components = {}                                            # output dictionary for dataframes
        self.not_matched = []                                           # grid objects without positions or errors
        # temp dictionary for converting data
        self._components = {key: dict() for key in ['nodes', 'limits', 'location_objects', 'layers',
                                                    'location_coords', 'transformers', 'energy_consumers',
                                                    'energy_generators']}

        self._load_components()

        self._geo_coder = GeoInformation()

    def _load_components(self):
        """
        convert and split the xml data in different dictionary to enable a better building process

        Parameters
        ----------
        :return:
        """

        locations, limits = defaultdict(list), {}

        for mRID, grid_object in self.grid_data.items():
            # all nodes which are included in the xml files
            if component_type(grid_object) == 'TopologicalNode':
                self._components['nodes'][mRID] = grid_object
                self._logger.info(f'add node: {mRID} to grid converter')
            # all transformers which are includes in the xml files
            elif component_type(grid_object) == 'PowerTransformer':
                self._components['transformers'][mRID] = grid_object
                self._logger.info(f'add transformer: {mRID} to grid converter')
            # all locations which are includes in the xml files
            elif component_type(grid_object) == 'Location':
                self._components['location_objects'][mRID] = grid_object
                self._logger.info(f'add location: {mRID} to grid converter')
            # temporary locations to get the coordinates
            elif component_type(grid_object) == 'PositionPoint':
                locations[grid_object.Location.mRID].append(grid_object)
            # current limit of all lines
            elif component_type(grid_object) == 'CurrentLimit':
                limits[mRID] = grid_object
                self._logger.info(f'add current limits: {mRID} to grid converter')
            # all consumer included in the xml files
            elif component_type(grid_object) == 'EnergyConsumer':
                self._components['energy_consumers'][mRID] = grid_object
                self._logger.info(f'add energy consumer: {mRID} to grid converter')
            elif component_type(grid_object) == 'SynchronousMachine':
                self._components['energy_generators'][mRID] = grid_object
                self._logger.info(f'add energy generator: {mRID} to grid converter')
        # get coordinates for each location and store them in self._components['location_coords']]
        for location_id, points in locations.items():
            x_coords, y_coords = [], []
            for point in points:
                x_coords.append(float(point.xPosition))
                y_coords.append(float(point.yPosition))
            self._components['location_coords'].update({location_id: {'lon_coords': x_coords, 'lat_coords': y_coords}})
        self._logger.info(f'build coordinates for {len(locations)}')
        # get current limits for each line to calculate further on the nominal power for each line
        for current_id, grid_object in limits.items():
            conducting_object_id = grid_object.OperationalLimitSet.Terminal.ConductingEquipment.mRID
            self._components['limits'].update({conducting_object_id: grid_object.value})
        self._logger.info(f'build current limits for {len(limits)}')

    def _build_nodes(self, voltage_level: float = 20):
        """
        Convert nodes in cgmes format to pandas dataframe
        with the columns: [id, v_nom, voltage_id, lon, lat, shape, injection]

        To determine the position, the algorithm do the following steps:
            1. check if the position can be determined by a consumer
            2. check if the position can be determined by a generator
            3. check if the position can be determined by more than one line
            4. check if the position can be determined by more one line

        In a last step the position is determined by the transformers, if that is possible

        Parameters
        ----------
        :param voltage_level: float
            maximal voltage level which is taken into account
        :return:
        """
        def get_point_from_power_system_resource(o):
            for location_id, location in self._components['location_objects'].items():
                if location.PowerSystemResources.mRID == o.mRID:
                    co = self._components['location_coords'][location_id]
                    return co['lon_coords'][0], co['lat_coords'][0]
            return None, None

        def get_point_from_equipment(o):
            try:
                coords = self._components['location_coords'][o.Location.mRID]
                return coords['lon_coords'], coords['lat_coords']
            except Exception as e:
                print(repr(e))
                return [None], [None]

        def get_connected_objects(n):
            connected2node = defaultdict(list)
            for terminal in n.Terminal:
                connected2node[component_type(terminal.ConductingEquipment)].append(terminal.ConductingEquipment)
            return connected2node

        def get_intersection(array):
            try:
                return max(set(array), key=array.count)
            except ValueError as e:
                print(repr(e))
                return None

        total_nodes = self._components['nodes']
        nodes = {}

        for node_id, grid_object in total_nodes.items():
            lon, lat = None, None
            not_found = True
            nominal_voltage = grid_object.BaseVoltage.nominalVoltage
            voltage_id = grid_object.BaseVoltage.mRID
            # check if node has any connection (terminal = connection)
            if isinstance(grid_object.Terminal, list) and nominal_voltage <= voltage_level:
                connected2node = get_connected_objects(grid_object)

                # check if an energy consumer has a position point (location)
                if 'EnergyConsumer' in connected2node.keys():
                    for element in connected2node['EnergyConsumer']:
                        lon, lat = get_point_from_power_system_resource(element)
                        if lon is not None and lat is not None:
                            self._logger.info('find point by energy consumer data')
                            not_found = False
                            break
                # check if a generator object has a position point (location)
                if 'SynchronousMachine' in connected2node.keys() and not_found:
                    for element in connected2node['SynchronousMachine']:
                        lon, lat = get_point_from_power_system_resource(element)
                        if lon is not None and lat is not None:
                            self._logger.info('find point by by synchronous generator data')
                            not_found = False
                            break
                # check if a position point can be determined by the lines
                elif 'ACLineSegment' in connected2node.keys() and not_found:
                    # for more than one line find the points and the intersection
                    if len(connected2node['ACLineSegment']) > 1:
                        lon_coords, lat_coords = [], []
                        # get the coordinates for each line
                        for line in connected2node['ACLineSegment']:
                            if line.Location:
                                lons, lats = get_point_from_equipment(line)
                            elif line.EquipmentContainer.Location:
                                lons, lats = get_point_from_equipment(line.EquipmentContainer)
                            else:
                                lons, lats = [None], [None]
                            lon_coords += lons
                            lat_coords += lats
                        # get the intersection
                        lon, lat = get_intersection(lon_coords), get_intersection(lat_coords)
                    # if one line is connected calculate the mean value
                    else:
                        line = connected2node['ACLineSegment'][0]
                        if line.Location:
                            lon, lat = get_point_from_equipment(line)
                            lon, lat = np.mean(lon), np.mean(lat)
                        elif line.EquipmentContainer.Location:
                            lon, lat = get_point_from_equipment(line.EquipmentContainer)
                            lon, lat = np.mean(lon), np.mean(lat)

            if lon is not None and lat is not None:
                nodes.update({node_id: {"v_nom": nominal_voltage, "lon": lon, "lat": lat, "shape": Point((lon, lat)),
                                        "voltage_id": voltage_id, 'injection': False}})
            elif isinstance(grid_object.Terminal, list):
                self.not_matched.append((node_id, grid_object))

        # check if the position can be determined by the transformers
        for node_id, node in self.not_matched:
            nominal_voltage = node.BaseVoltage.nominalVoltage
            if isinstance(node.Terminal, list) and nominal_voltage <= voltage_level:
                connected2node = get_connected_objects(node)
                # get power transformers
                if 'PowerTransformer' in connected2node.keys():
                    for power_transformer in connected2node['PowerTransformer']:
                        connected = power_transformer.PowerTransformerEnd[0].Terminal.TopologicalNode.mRID
                        if connected in nodes.keys():
                            lon, lat = nodes[connected]['lon'], nodes[connected]['lat']
                            nodes.update({node_id: {"v_nom": nominal_voltage, "lon": lon, "lat": lat,
                                                    "shape": Point((lon, lat)),  "voltage_id": node.BaseVoltage.mRID,
                                                    "injection": False}})
                            break

        self.components['nodes'] = pd.DataFrame.from_dict(nodes, orient='index').dropna()

    def _build_edges(self):
        """
        Convert edges in cgmes format to pandas dataframe
        with the columns: [id, bus0, bus1, s_nom, r, x, len,voltage_id, lon_coords, lat_coords, shape]

        The position is directly determined over the location id of the line or by its EquipmentContainer

        Parameters
        ----------
        :return:
        """
        edges = {}
        # for all nodes find lines and positions
        for index in self.components['nodes'].index:
            for terminal in self._components['nodes'][index].Terminal:
                object = terminal.ConductingEquipment

                if component_type(object) == 'ACLineSegment':
                    if object.mRID not in edges.keys():
                        if object.Location:
                            location_id = object.Location.mRID
                        else:
                            location_id = object.EquipmentContainer.Location.mRID

                        coords = self._components['location_coords'][location_id]
                        lon_coords, lat_coords = coords['lon_coords'], coords['lat_coords']
                        points = [(lon_coords[i], lat_coords[i]) for i in range(len(lon_coords))]
                        shape = LineString([Point(point) for point in points])

                        nominal_voltage = self._components['nodes'][index].BaseVoltage.nominalVoltage
                        nominal_current = self._components['limits'][object.mRID]

                        edges.update({object.mRID: {'bus0': index, 'bus1': None,
                                                    'lon_coords': lon_coords, 'lat_coords': lat_coords,
                                                    'shape': shape, 'r': object.r, 'x': object.x,
                                                    's_nom': 3 ** (1 / 2) * nominal_voltage * nominal_current / 1_000,
                                                    'len': object.length, 'v_nom': nominal_voltage}})

                    elif object.mRID in edges.keys():
                        edges[object.mRID]['bus1'] = index

        self.components['edges'] = pd.DataFrame.from_dict(edges, orient='index')

    def _build_transformer(self):
        """
        Convert transformers in cgmes format to pandas dataframe
        with the columns: [id, bus0, v0, bus1, v1, s_nom, r, x, b, g voltage_id, lat, lon, shape]

        The position is directly determined over the connected node.
        Additionally the property injection of the node is set to True, if that node has transformer

        Parameters
        ----------
        :return:
        """
        transformers = {}
        for transformer_id, grid_object in self._components['transformers'].items():
            # low voltage level
            bus0 = grid_object.PowerTransformerEnd[1]
            bus0_voltage = bus0.BaseVoltage.nominalVoltage
            bus0_id = bus0.Terminal.TopologicalNode.mRID
            # high voltage level
            bus1 = grid_object.PowerTransformerEnd[0]
            bus1_voltage = bus1.BaseVoltage.nominalVoltage
            bus1_id = bus1.Terminal.TopologicalNode.mRID

            if bus0_id in self.components['nodes'].index:
                lon, lat = self.components['nodes'].loc[bus0_id, 'lon'], self.components['nodes'].loc[bus0_id, 'lat']
                shape = Point(lon, lat)
                transformers.update({transformer_id: {
                    'bus0': bus0_id, 'v0': bus0_voltage,
                    'bus1': bus1_id, 'v1': bus1_voltage,
                    'voltage_id': bus0.BaseVoltage.mRID, 'r': bus1.r, 'x': bus1.x, 'b': bus1.b, 'g': bus1.g,
                    's_nom': bus0.ratedS, 'lon': lon, 'lat': lat, 'shape': shape}})
                self.components['nodes'].loc[bus0_id, 'injection'] = True

        self.components['transformers'] = pd.DataFrame.from_dict(transformers, orient='index')


    def _build_consumers(self):
        """
        Convert energy consumer in cgmes format to pandas dataframe
        with the columns: []

        The position is directly determined over the connected node.

        Parameters
        ----------
        :return:
        """
        consumers = {}
        for mRID, consumer in self._components['energy_consumers'].items():
            container = consumer.EquipmentContainer
            nominal_voltage = container.BaseVoltage.nominalVoltage
            if isinstance(container.TopologicalNode, list):
                node_id = container.TopologicalNode[0].mRID
                if node_id in self.components['nodes'].index:
                    consumers[mRID] = {'bus0': node_id, 'v_nom': nominal_voltage,
                                      'lat': self.components['nodes'].loc[node_id, 'lat'],
                                      'lon': self.components['nodes'].loc[node_id, 'lon'],
                                      'shape': self.components['nodes'].loc[node_id, 'shape']
                                      }

        self.components['consumers'] = pd.DataFrame.from_dict(consumers, orient='index')

    def _build_generators(self):
        """
        Convert energy generator in cgmes format to pandas dataframe
        with the columns: []

        The position is directly determined over the connected node.

        Parameters
        ----------
        :return:
        """
        generators = {}
        for mRID, generator in self._components['energy_generators'].items():
            container = generator.EquipmentContainer
            nominal_voltage = container.BaseVoltage.nominalVoltage
            if isinstance(container.TopologicalNode, list):
                node_id = container.TopologicalNode[0].mRID
                if node_id in self.components['nodes'].index:
                    generators[mRID] = {'bus0': node_id, 'v_nom': nominal_voltage,
                                       'lat': self.components['nodes'].loc[node_id, 'lat'],
                                       'lon': self.components['nodes'].loc[node_id, 'lon'],
                                       'shape': self.components['nodes'].loc[node_id, 'shape'],
                                       'p_set':  np.abs(generator.p) * 1000
                                      }

        self.components['generators'] = pd.DataFrame.from_dict(generators, orient='index')

    def _build_geo_info(self):
        coords = self.components['consumer'].loc[self.components['consumers']['v_nom'] == 0.4, ['lon', 'lat']].values
        self._geo_coder.poi_s = list(set([(coords[i][0], coords[i][1]) for i in range(len(coords))]))
        for feature, point in self._geo_coder.get_information():
            if point is not None:
                self._components['layers'][(point.x, point.y)] = dict(source=feature, type='fill',
                                                                      below='traces', color='rgba(33,47,61,0.3)')

    def convert(self, voltage_level: float = 20):
        """

        Parameters
        ----------
        :return:
        """
        try:
            self._build_nodes(voltage_level=voltage_level)
            self._build_edges()
            self._build_transformer()
            self._build_consumers()
            self._build_generators()
            # self._build_geo_info()
            self._logger.info('conversion complete -> dataframes are accessible')
        except Exception as e:
            self._logger.error('Error while conversion')
            print(repr(e))

    def save(self, path: str = r'./data/export/'):
        try:
            self.components['nodes'].to_csv(f'{path}nodes.csv')
            self.components['edges'].to_csv(f'{path}edges.csv')
            self.components['transformers'].to_csv(f'{path}transformers.csv')
            self.components['consumers'].to_csv(f'{path}consumers.csv')
            self.components['generators'].to_csv(f'{path}generators.csv')
            with open(f'{path}layers.pkl', 'wb') as handle:
                pickle.dump(self._components['layers'], handle, protocol=pickle.HIGHEST_PROTOCOL)

        except Exception as e:
            self._logger.error('Error while saving the files')
            print(repr(e))

    def plot(self):
        """
        Plot the resulting grid model with a plotly graphic object. The figure is shown in browser via html file
        It displays each voltage levels in an own trace containing nodes, lines, transformers and consumer.

        Parameters
        ----------
        :return:
        """
        try:
            show_figure(self.components['nodes'], self.components['edges'],
                        self.components['transformers'],
                        self.components['consumers'].loc[self.components['consumers']['v_nom']==0.4],
                        layers=[])
        except Exception as e:
            self._logger.error('cant plot data')
            print(repr(e))

if __name__ == "__main__":
    converter = CGMESConverter()
    converter.convert()
    converter.save()
    # converter.plot()


