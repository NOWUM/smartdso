import logging
import os
import pandas as pd
import numpy as np
from datetime import timedelta as td, datetime
from sqlalchemy import create_engine


from gridLib.model import GridModel
logging.getLogger('pypsa').setLevel('ERROR')

# -> pandas frequency names
RESOLUTION = {1440: 'min', 96: '15min', 24: 'h'}
# -> database uri to store the results
DATABASE_URI = os.getenv('DATABASE_URI', 'postgresql://opendata:opendata@10.13.10.41:5432/smartdso')

logger = logging.getLogger('CapacityProvider')


class CapacityProvider:

    def __init__(self, scenario: str, iteration: int, start_date: datetime, end_date: datetime,
                 database_uri: str = DATABASE_URI, T: int = 1440, write_geo: bool = True, *args, **kwargs):
        self.scenario = scenario
        self.iteration = iteration
        # -> build grid model and set simulation horizon
        self.grid = GridModel()
        self.mapper = self.grid.model.buses['sub_network']
        self.mapper = self.mapper[self.mapper.index.isin(self.grid.data['connected'].index)]

        self.time_range = pd.date_range(start=start_date, end=end_date + td(days=1), freq=RESOLUTION[T])[:-1]

        self.line_utilization = {sub_id: pd.DataFrame(columns=self.grid.sub_networks[sub_id]['model'].lines.index,
                                                      index=self.time_range)
                                 for sub_id in self.mapper.unique()}

        self.transformer_utilization = {sub_id: pd.DataFrame(columns=['utilization'], index=self.time_range)
                                        for sub_id in self.mapper.unique()}
        self.demand = pd.DataFrame()

        self._rq_l_util = pd.DataFrame()
        self._rq_t_util = pd.DataFrame()

        self.T = T
        self._database = create_engine(database_uri)

        self._geo_info = dict(edges=[], nodes=[], transformers=[])

        for sub_id in self.mapper.unique():
            for asset_type in self._geo_info.keys():
                df = self.grid.get_components(asset_type, grid=sub_id)
                df.set_crs(crs='EPSG:4326', inplace=True)
                self._geo_info[asset_type] += [df]
                if write_geo:
                    df['asset'] = asset_type
                    # df.set_index('name', inplace=True)
                    df.to_postgis(name=f'{asset_type}_geo', con=self._database, if_exists='append')

    def _line_utilization(self, sub_id: str) -> pd.DataFrame:
        lines = self.grid.sub_networks[sub_id]['model'].lines_t.p0
        s_max = self.grid.sub_networks[sub_id]['model'].lines.loc[:, 's_nom']
        for column in lines.columns:
            lines[column] = np.abs(lines[column]) / s_max.loc[column] * 100
        return lines

    def _transformer_utilization(self, sub_id: str) -> pd.DataFrame:
        transformer = self.grid.sub_networks[sub_id]['model'].generators_t.p
        s_max = self.grid.sub_networks[sub_id]['s_max']
        transformer = transformer/s_max * 100
        return transformer

    def run_power_flow(self, data: pd.DataFrame, sub_id: int) -> None:
        demand_data = data.loc[data.index.get_level_values(0).isin(self.mapper[self.mapper == sub_id].index)]
        demand_data = demand_data.reset_index().drop_duplicates(subset=['node_id', 'power'])
        snapshots = list(demand_data['t'].unique())
        self.grid.sub_networks[sub_id]['model'].set_snapshots(snapshots)
        for node in demand_data['node_id'].unique():
            # -> demand time series in [kW] to [MW]
            demand = demand_data.loc[demand_data['node_id'] == node, ['t', 'power']]
            demand['power'] /= 1000
            demand = demand.set_index('t')
            self.grid.sub_networks[sub_id]['model'].loads_t.p_set[f'{node}_consumer'] = demand
        self.grid.sub_networks[sub_id]['model'].loads_t.p_set.fillna(method='ffill', inplace=True)
        self.grid.run_power_flow(sub_id=sub_id)

    def set_fixed_power(self, data: pd.DataFrame) -> None:
        data = data.groupby(['node_id', 't']).sum()
        self.demand = data.copy()
        for sub_id in self.mapper.unique():
            self.run_power_flow(data=data, sub_id=sub_id)
            snapshots = self.grid.sub_networks[sub_id]['model'].snapshots
            self._rq_l_util = self._line_utilization(sub_id=sub_id)
            self._rq_t_util = self._transformer_utilization(sub_id=sub_id)

            self.line_utilization[sub_id].loc[snapshots, self._rq_l_util.columns] = self._rq_l_util.values
            self.line_utilization[sub_id].fillna(method='ffill', inplace=True)
            self.transformer_utilization[sub_id].loc[snapshots, 'utilization'] = self._rq_t_util.values.flatten()
            self.transformer_utilization[sub_id].fillna(method='ffill', inplace=True)

    def get_price(self, request: pd.Series = None, node_id: str = '') -> pd.Series:
        def price_func(util):
            if util > 100:
                return 9_999
            else:
                return ((-np.log(1 - np.power(util / 100, 1.5)) + 0.175) * 0.15) * 100

        # -> set current demand at each node
        data = self.demand.copy()
        data = data.loc[data.index.get_level_values(level='t').isin(request[request.values > 0].index)]
        request_ = pd.DataFrame(request)
        request_.columns = ['power']
        request_ = request_.rename_axis('t').reset_index()
        request_['node_id'] = node_id
        request_.set_index(['node_id', 't'], inplace=True)

        data = pd.concat([data, request_], axis=0)
        data = data.groupby(['node_id', 't']).sum()

        sub_id = self.mapper[node_id]
        self.run_power_flow(data=data, sub_id=sub_id)
        self._rq_l_util = self._line_utilization(sub_id=sub_id)
        self._rq_t_util = self._transformer_utilization(sub_id=sub_id)

        utilization = pd.concat([self._rq_l_util, self._rq_t_util,
                                 self.line_utilization[sub_id].loc[request.index],
                                 self.transformer_utilization[sub_id].loc[request.index]], axis=1).max(axis=1)
        response = pd.Series(index=request.index, data=[price_func(u) for u in utilization.values])

        return response

    def commit(self, request: pd.Series, node_id: str) -> None:
        sub_id = self.mapper[node_id]
        self.demand.loc[(self.demand.index.get_level_values(0) == node_id) &
                        (self.demand.index.get_level_values(1).isin(request.index)), 'power'] += request.values

        snapshots = self.grid.sub_networks[sub_id]['model'].snapshots
        self.line_utilization[sub_id].loc[snapshots, self._rq_l_util.columns] = self._rq_l_util.values
        self.line_utilization[sub_id].fillna(method='ffill', inplace=True)
        self.transformer_utilization[sub_id].loc[snapshots, 'utilization'] = self._rq_t_util.values.flatten()
        self.transformer_utilization[sub_id].fillna(method='ffill', inplace=True)

    def save_results(self, d_time: datetime):

        time_range = pd.date_range(start=d_time, freq=RESOLUTION[self.T], periods=self.T)

        for sub_id, dataframe in self.line_utilization.items():
            for column in dataframe.columns:
                if self.grid.model.lines.loc[column, 'bus0'] in self.grid.data['connected'].index \
                        or self.grid.model.lines.loc[column, 'bus1'] in self.grid.data['connected'].index:
                    asset = 'outlet'
                else:
                    asset = 'inlet'

                df = pd.DataFrame(dict(time=time_range, iteration=self.T * [int(self.iteration)],
                                       scenario=self.T * [self.scenario], id_=self.T * [column],
                                       sub_grid=self.T * [int(sub_id)], asset=self.T * [asset],
                                       utilization=dataframe.loc[time_range, column].values))

                df = df.loc[df['utilization'] == df['utilization'].max(), :]

                try:
                    df.to_sql('grid', self._database, index=False, if_exists='append')
                except Exception as e:
                    logger.warning(f'server closed the connection {repr(e)}')
                    df.to_sql('grid', self._database, index=False, if_exists='append')
                    logger.error(f'data for line {column} are not stored in database')

        for sub_id, dataframe in self.transformer_utilization.items():
            id_ = int(sub_id)
            df = pd.DataFrame(dict(time=time_range,
                                   iteration=self.T * [int(self.iteration)],
                                   scenario=self.T * [self.scenario],
                                   id_=self.T * [self._geo_info['transformers'][id_]['name'].values[0]],
                                   sub_grid=self.T * [int(sub_id)],
                                   asset=self.T * ['transformer'],
                                   utilization=dataframe.loc[time_range, 'utilization'].values))
            try:
                df.to_sql('grid', self._database, index=False, if_exists='append')
            except Exception as e:
                logger.warning(f'server closed the connection {repr(e)}')
                df.to_sql('grid', self._database, index=False, if_exists='append')
                logger.error(f'data for transformer {sub_id} are not stored in database')


if __name__ == "__main__":
    cp = CapacityProvider(**dict(start_date=datetime(2022, 1, 1), end_date=datetime(2022, 1, 2),
                                 scenario=None, iteration=None, T=96), write_geo=True)

