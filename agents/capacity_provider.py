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
                 database_uri: str = DATABASE_URI, T: int = 1440, write_geo: bool = True,
                 sub_grid: int = -1, *args, **kwargs):
        self.scenario = scenario
        self.iteration = iteration
        # -> build grid model and set simulation horizon
        self.grid = GridModel()
        self.mapper = self.grid.model.buses['sub_network']
        self.mapper = self.mapper[self.mapper.index.isin(self.grid.data['connected'].index)]
        # -> get valid sub grid, which are build successfully
        valid_sub_grid = [value in self.grid.sub_networks.keys() for value in self.mapper.values]
        self.mapper = self.mapper.loc[valid_sub_grid]
        self.sub_grid = sub_grid
        if self.sub_grid != -1:
            self.sub_ids = [str(self.sub_grid)]
        else:
            self.sub_ids = self.mapper.unique()

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
        transformer = transformer / s_max * 100
        return transformer

    def run_power_flow(self, data: pd.DataFrame, sub_id: str, end_of_day: bool = False,
                       d_time: datetime = None) -> None:
        nodes_in_grid = self.mapper[self.mapper.values == sub_id].index
        demand_data = data.loc[data.index.get_level_values('node_id').isin(nodes_in_grid)]
        demand_data = demand_data.reset_index()
        if end_of_day:
            snapshots = list(pd.date_range(start=d_time, periods=self.T, freq=RESOLUTION[self.T]))
        else:
            demand_unique = demand_data.drop_duplicates(subset=['node_id', 'power'])
            snapshots = list(demand_unique['t'].unique())

        self.grid.sub_networks[sub_id]['model'].set_snapshots(snapshots)
        for node in demand_data['node_id'].unique():
            # -> demand time series in [kW] to [MW]
            demand = demand_data.loc[demand_data['node_id'] == node, ['t', 'power']]
            demand['power'] /= 1000
            demand = demand.set_index('t')
            self.grid.sub_networks[sub_id]['model'].loads_t.p_set[f'{node}_consumer'] = demand.loc[snapshots]
        self.grid.run_power_flow(sub_id=sub_id)

    def set_fixed_power(self, data: pd.DataFrame) -> None:
        self.demand = data.groupby(['node_id', 't']).sum()
        for sub_id in self.sub_ids:
            for date in np.unique(self.time_range.date):
                self.run_power_flow(data=self.demand.copy(), sub_id=sub_id, end_of_day=True, d_time=date)

                snapshots = self.grid.sub_networks[sub_id]['model'].snapshots

                self._rq_l_util = self._line_utilization(sub_id=sub_id)
                self._rq_t_util = self._transformer_utilization(sub_id=sub_id)

                self.line_utilization[sub_id].loc[snapshots, self._rq_l_util.columns] = self._rq_l_util.values
                self.transformer_utilization[sub_id].loc[snapshots, 'utilization'] = self._rq_t_util.values.flatten()

    def get_price(self, request: pd.Series = None, node_id: str = '') -> pd.Series:
        def price_func(util):
            if util > 100:
                return 9_999
            else:
                return ((-np.log(1 - np.power(util / 100, 1.5)) + 0.175) * 0.15) * 100

        # -> set current demand at each node
        data = self.demand.copy()
        data = data.loc[data.index.get_level_values(level='t').isin(request.index)]
        request_ = pd.DataFrame(request)
        request_.columns = ['power']
        request_ = request_.rename_axis('t').reset_index()
        request_['node_id'] = node_id
        request_.set_index(['node_id', 't'], inplace=True)

        data = pd.concat([data, request_], axis=0)
        data = data.groupby(['node_id', 't']).sum()
        data = data.sort_index(level='t')

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
        node = self.demand.index.get_level_values(0) == node_id
        time = self.demand.index.get_level_values(1).isin(request.index)
        self.demand.loc[node & time, 'power'] += request.values

        # snapshots = self.grid.sub_networks[sub_id]['model'].snapshots
        # self.line_utilization[sub_id].loc[snapshots, self._rq_l_util.columns] = self._rq_l_util.values
        # self.transformer_utilization[sub_id].loc[snapshots, 'utilization'] = self._rq_t_util.values.flatten()

    def _save_summary(self, d_time: datetime) -> None:

        time_range = pd.date_range(start=d_time, freq=RESOLUTION[self.T], periods=self.T)

        aggregate_functions = {'mean': lambda x: pd.DataFrame.mean(x, axis=1),
                               'max': lambda x: pd.DataFrame.max(x, axis=1),
                               'median': lambda x: pd.DataFrame.median(x, axis=1)}

        def build_data(d: pd.DataFrame, asset: str = 'line', s_id: int = 0, tp: str = 'mean'):
            d.columns = ['value']
            d['type'] = tp
            d['sub_id'] = s_id
            d['asset'] = asset
            d['scenario'] = self.scenario
            d['iteration'] = self.iteration
            d.index.name = 'time'

            d = d.reset_index()
            d = d.set_index(['time', 'scenario', 'iteration', 'type', 'asset', 'sub_id'])

            return d

        for sub_id in self.sub_ids:
            dataframe = self.line_utilization[sub_id]
            for key, function in aggregate_functions.items():
                data = pd.DataFrame(function(dataframe.loc[time_range, :]))
                data = build_data(data, asset='line', s_id=int(sub_id), tp=key)
                try:
                    data.to_sql('grid_summary', self._database, if_exists='append', method='multi')
                except Exception as e:
                    logger.warning(f'server closed the connection {repr(e)}')

        for sub_id in self.sub_ids:
            dataframe = self.transformer_utilization[sub_id]
            for key, function in aggregate_functions.items():
                data = pd.DataFrame(function(dataframe.loc[time_range, :]))
                data = build_data(data, asset='transformer', s_id=int(sub_id), tp=key)

                try:
                    data.to_sql('grid_summary', self._database, if_exists='append', method='multi')
                except Exception as e:
                    logger.warning(f'server closed the connection {repr(e)}')

    def _save_grid_asset(self, d_time: datetime) -> None:

        time_range = pd.date_range(start=d_time, freq=RESOLUTION[self.T], periods=self.T)

        for sub_id in self.sub_ids:
            dataframe = self.line_utilization[sub_id]
            for line in dataframe.columns:
                result = dataframe.loc[time_range, [line]]
                result.columns = ['utilization']
                result['id_'] = line
                result['asset'] = 'line'
                result['scenario'] = self.scenario
                result['iteration'] = self.iteration
                result['sub_id'] = int(sub_id)
                result.index.name = 'time'
                result = result.reset_index()
                result = result.set_index(['time', 'scenario', 'iteration', 'id_'])
                try:
                    result.to_sql('grid_asset', self._database, if_exists='append', method='multi')
                except Exception as e:
                    logger.warning(f'server closed the connection {repr(e)}')

        for sub_id in self.sub_ids:
            result = self.transformer_utilization[sub_id].loc[time_range, ['utilization']]
            result['id_'] = self.grid.get_components('transformers', sub_id).name.values[0]
            result['asset'] = 'transformer'
            result['scenario'] = self.scenario
            result['iteration'] = self.iteration
            result['sub_id'] = int(sub_id)
            result.index.name = 'time'
            result = result.reset_index()
            result = result.set_index(['time', 'scenario', 'iteration', 'id_'])
            try:
                result.to_sql('grid_asset', self._database, if_exists='append', method='multi')
            except Exception as e:
                logger.warning(f'server closed the connection {repr(e)}')

    def _run_end_of_day(self, d_time: datetime):
        for sub_id in self.sub_ids:
            self.run_power_flow(data=self.demand.copy(), sub_id=sub_id, end_of_day=True, d_time=d_time)
            self._rq_l_util = self._line_utilization(sub_id=sub_id)
            self._rq_t_util = self._transformer_utilization(sub_id=sub_id)
            time_range = pd.date_range(start=d_time, periods=self.T, freq=RESOLUTION[self.T])
            self.line_utilization[sub_id].loc[time_range, self._rq_l_util.columns] = self._rq_l_util.values
            self.transformer_utilization[sub_id].loc[time_range, 'utilization'] = self._rq_t_util.values.flatten()

    def save_results(self, d_time: datetime) -> None:
        # self._run_end_of_day(d_time)
        self._save_summary(d_time)
        if self.iteration == 0:
            self._save_grid_asset(d_time)


if __name__ == "__main__":
    cp = CapacityProvider(**dict(start_date=datetime(2022, 1, 1), end_date=datetime(2022, 1, 2),
                                 scenario=None, iteration=None, T=96), write_geo=True)
