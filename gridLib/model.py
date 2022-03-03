import numpy as np
import pandas as pd
import pypsa
import logging
# from shapely.wkt import loads

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

class MissingFile(Exception): pass
class NoGridsIncluded(Exception): pass


class GridModel:

    def __init__(self):

        self._logger = logging.getLogger('GridModel')
        self._logger.setLevel('ERROR')
        self._data_path = r'./gridLib/data/export'
        self._T = 1440

        self.data = {}
        self.model = pypsa.Network()

        self._load_grid_data()
        self._logger.info(f'start building network for data in folder: {self._data_path}')
        self._generate_grid_model()

    def _load_grid_data(self):

        grid_data = {
            'nodes': pd.read_csv(fr'{self._data_path}/nodes.csv', index_col=0),
            'transformers': pd.read_csv(fr'{self._data_path}/transformers.csv', index_col=0),
            'edges': pd.read_csv(fr'{self._data_path}/edges.csv', index_col=0),
            'consumers': pd.read_csv(fr'{self._data_path}/grid_allocations.csv', index_col=0)
        }
        grid_data['connected'] = grid_data['nodes'].loc[grid_data['nodes'].index.isin(grid_data['consumers']['bus0'])]
        grid_data['voltage_ids'] = grid_data['connected']['voltage_id'].unique()

        self.power_series = {node: np.zeros(self._T) for node in grid_data['connected'].index}

        self.data = grid_data

    def _generate_grid_model(self):

        # add busses to network --> each node is a bus
        nodes = self.data['nodes'].loc[self.data['nodes']['voltage_id'].isin(self.data['voltage_ids'])]
        self.model.madd('Bus', names=nodes.index, v_nom=nodes.v_nom, x=nodes.lon, y=nodes.lat,)

        # add edges to network to connect the busses
        edges = self.data['edges']
        edges = edges.loc[edges['bus0'].isin(nodes.index) | edges['bus1'].isin(nodes.index)]
        self.model.madd('Line', names=edges.index, bus0=edges.bus0, bus1=edges.bus1, x=edges.x, r=edges.r,
                        s_nom=edges.s_nom)
        # add slack generators for each transformer with higher voltage
        transformers = self.data['transformers']
        transformers = transformers.loc[transformers['bus0'].isin(nodes.index) | transformers['bus1'].isin(nodes.index)]
        slack_nodes = transformers['bus0'].unique()
        for node in slack_nodes:
            self.model.add('Generator', name=f'{node}_slack', bus=node, control='Slack')

        for node, consumer in self.data['connected'].iterrows():
            self.model.add('Load', name=f'{node}_consumer', bus=node)

        self.model.determine_network_topology()
        self.model.consistency_check()
        self.model.snapshots = pd.Series(range(self._T))

    def run_power_flow(self, snapshots=None, fixed=True):

        for node in self.power_series.keys():
            self.model.loads_t.p_set[f'{node}_consumer'] = self.power_series[node]/1000
        try:
            if fixed:
                result_summary = self.model.pf(snapshots=[i for i in np.arange(0, 1440, 15)])
            else:
                result_summary = self.model.pf(snapshots)
            converged = result_summary.converged
            for col in converged.columns:
                if any(converged[col]):
                    self._logger.warning(f'pfc for subgrid {col} not converged')
                else:
                    self._logger.info(f'pfc for subgrid {col} converged')

        except Exception as e:
            self._logger.error(repr(e))
            self._logger.error('error during power flow calculation')




