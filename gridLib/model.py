import pandas as pd
import pypsa
import logging
from shapely.wkt import loads
import geopandas as gpd

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

data_path = r'./gridLib/data/export'
# -> read nodes
total_nodes = pd.read_csv(fr'{data_path}/nodes.csv', index_col=0)
total_nodes['geometry'] = total_nodes['shape'].apply(loads)
# -> read transformers
total_transformers = pd.read_csv(fr'{data_path}/transformers.csv', index_col=0)
total_transformers['geometry'] = total_transformers['shape'].apply(loads)
# -> read edges
total_edges = pd.read_csv(fr'{data_path}/edges.csv', index_col=0)
total_edges['geometry'] = total_edges['shape'].apply(loads)
# -> read known consumers
total_consumers = pd.read_csv(fr'{data_path}/grid_allocations.csv', index_col=0)
# total_consumers = pd.read_csv(fr'{data_path}/consumers.csv', index_col=0)

class GridModel:

    def __init__(self):
        logging.getLogger('pypsa').setLevel('ERROR')
        # --> set logger
        self._logger = logging.getLogger('GridModel')
        self._logger.setLevel('ERROR')

        self.sub_networks = []

        self.model = pypsa.Network()   # --> total grid model
        self.sub_networks = {}         # --> dict for sub network models

        # --> build grid data
        grid_data = {'nodes': total_nodes, 'transformers': total_transformers, 'edges': total_edges,
                     'consumers': total_consumers}
        grid_data['connected'] = grid_data['nodes'].loc[grid_data['nodes'].index.isin(grid_data['consumers']['bus0'])]
        grid_data['voltage_ids'] = grid_data['connected']['voltage_id'].unique()
        self.data = grid_data.copy()

        # --> add busses to network --> each node is a bus
        nodes = self.data['nodes'].loc[self.data['nodes']['voltage_id'].isin(self.data['voltage_ids'])]
        self.model.madd('Bus', names=nodes.index, v_nom=nodes.v_nom, x=nodes.lon, y=nodes.lat)

        # --> add edges to network to connect the busses
        edges = self.data['edges']
        edges = edges.loc[edges['bus0'].isin(nodes.index) | edges['bus1'].isin(nodes.index)]
        self.model.madd('Line', names=edges.index, bus0=edges.bus0, bus1=edges.bus1, x=edges.x, r=edges.r,
                        s_nom=edges.s_nom)

        # --> add slack generators for each transformer with higher voltage
        transformers = self.data['transformers']
        transformers = transformers.loc[transformers['bus0'].isin(nodes.index) | transformers['bus1'].isin(nodes.index)]
        slack_nodes = transformers['bus0'].unique()
        for node in slack_nodes:
            self.model.add('Generator', name=f'{node}_slack', bus=node, control='Slack')

        # --> add consumers to node
        for node, consumer in self.data['connected'].iterrows():
            self.model.add('Load', name=f'{node}_consumer', bus=node)

        # --> determine sub networks and check consistency
        self.model.determine_network_topology()
        self.model.consistency_check()

        for index in self.model.sub_networks.index:
            model = pypsa.Network()
            nodes = self.model.buses.loc[self.model.buses['sub_network'] == index]
            model.madd('Bus', names=nodes.index, v_nom=nodes.v_nom, x=nodes.x, y=nodes.y)
            lines = self.model.lines.loc[self.model.lines['sub_network'] == index]
            model.madd('Line', names=lines.index, bus0=lines.bus0, bus1=lines.bus1, x=lines.x, r=lines.r,
                       s_nom=lines.s_nom)

            generators = nodes['generator'].dropna()
            name = generators.values[0]
            generator = self.model.generators.loc[name]
            model.add('Generator', name=name, bus=generator.bus, control='Slack')
            consumers = self.model.loads.loc[self.model.loads['bus'].isin(nodes.index)]
            model.madd('Load', names=consumers.index, bus=consumers.bus)

            model.determine_network_topology()
            model.consistency_check()

            self.sub_networks[index] = model

    def get_components(self, type_: str = 'edges', grid='total'):
        if grid == 'total':
            model = self.model
        else:
            model = self.sub_networks[str(grid)]

        if type_ == 'nodes':
            data = self.data[type_].loc[self.data[type_].index.isin(model.buses.index)]
        elif type_ == 'edges':
            data = self.data[type_].loc[self.data[type_].index.isin(model.lines.index)]
        else:
            i = self.data[type_]['bus0'].isin(model.buses.index) | self.data[type_]['bus1'].isin(model.buses.index)
            data = self.data[type_].loc[i]
        data.index.EV_RATIO = 'name'
        df = data.reset_index()
        return gpd.GeoDataFrame(df.loc[:, ['name', 'geometry']], geometry='geometry')

    def run_power_flow(self, sub_id: int):

        try:
            result_summary = self.sub_networks[sub_id].pf()
            converged = result_summary.converged
            for col in converged.columns:
                if any(converged[col]):
                    self._logger.warning(f'pfc for subgrid {col} not converged')
                else:
                    self._logger.info(f'pfc for subgrid {col} converged')

        except Exception as e:
            self._logger.error(repr(e))
            self._logger.error('error during power flow calculation')

if __name__ == "__main__":
    from plotting import show_plot
    import plotly
    key = 'pk.eyJ1Ijoicmlla2VjaCIsImEiOiJjazRiYTdndXkwYnN3M2xteGN2MHhtZjB0In0.33tSDK45TXF3lb3-G147jw'
    # plotly.io.orca.config.mapbox_access_token = key
    m = GridModel()
    # for s_id in [f'{i}' for i in range(5)]:
    #     index_set = m.sub_networks[s_id].buses.index
    #     nodes = m.data['nodes'].loc[index_set]
    #     edges = m.data['edges'].loc[m.data['edges']['bus0'].isin(index_set)
    #                                 | m.data['edges']['bus1'].isin(index_set)]
    #     transformers = m.data['transformers'].loc[m.data['transformers']['bus0'].isin(index_set)
    #                                               | m.data['transformers']['bus1'].isin(index_set)]
    #     # consumers = m.data['connected']
    #     consumers = pd.DataFrame()
    #
    #     fig =show_plot(nodes=nodes, edges=edges, transformers=transformers, consumers=consumers)
    #     font = dict(family="Verdana", size=10, color="black")
    #     fig.update_layout(font=font, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    #     fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="right", x=1))
    #     fig.write_image(rf'./images/sub_{s_id}.svg', width=1200, height=600)


