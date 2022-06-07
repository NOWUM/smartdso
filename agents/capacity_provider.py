import logging
import pandas as pd
import numpy as np
from datetime import timedelta as td, datetime


from gridLib.model import GridModel
logging.getLogger('pypsa').setLevel('ERROR')


class CapacityProvider:

    def __init__(self, *args, **kwargs):
        self.scenario = kwargs['scenario']
        self.iteration = kwargs['iteration']
        # -> build grid model and set simulation horizon
        self.grid = GridModel()
        self.mapper = self.grid.model.buses['sub_network']
        self.mapper = self.mapper[self.mapper.index.isin(self.grid.data['connected'].index)]

        self.time_range = pd.date_range(start=pd.to_datetime(kwargs['start_date']),
                                        end=pd.to_datetime(kwargs['end_date']) + td(days=1), freq='min')[:-1]

        self.line_utilization = {sub_id: pd.DataFrame(columns=self.grid.sub_networks[sub_id]['model'].lines.index,
                                                      index=self.time_range)
                                 for sub_id in self.mapper.unique()}

        self.transformer_utilization = {sub_id: pd.DataFrame(columns=['utilization'], index=self.time_range)
                                        for sub_id in self.mapper.unique()}
        self.demand = pd.DataFrame()

        self._rq_l_util = pd.DataFrame()
        self._rq_t_util = pd.DataFrame()

    def _line_utilization(self, sub_id: str):
        lines = self.grid.sub_networks[sub_id]['model'].lines_t.p0
        s_max = self.grid.sub_networks[sub_id]['model'].lines.loc[:, 's_nom']
        for column in lines.columns:
            lines[column] = np.abs(lines[column]) / s_max.loc[column] * 100
        return lines

    def _transformer_utilization(self, sub_id: str):
        transformer = self.grid.sub_networks[sub_id]['model'].generators_t.p
        s_max = self.grid.sub_networks[sub_id]['s_max']
        transformer = transformer/s_max * 100
        return transformer

    def run_power_flow(self, data: pd.DataFrame, sub_id: int):
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

    def set_fixed_power(self, data: pd.DataFrame):
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

    def get_price(self, request: pd.Series = None, node_id: str = ''):
        def price_func(util):
            if util > 100:
                return 9_999
            else:
                return ((-np.log(1 - np.power(util / 100, 1.5)) + 0.175) * 0.15) * 100

        # -> set current demand at each node
        data = self.demand.copy()
        data = data.loc[(data.index.get_level_values(level='t') <= request.index[-1]) &
                        (data.index.get_level_values(level='t') >= request.index[0])]
        request_ = pd.DataFrame(data=dict(node_id=len(request) * [node_id], t=request.index, power=request.values))
        request_.set_index(['node_id', 't'], inplace=True)

        data = pd.concat([data, request_], axis=0)
        data = data.groupby(['node_id', 't']).sum()

        sub_id = self.mapper[node_id]
        self.run_power_flow(data=data, sub_id=sub_id)
        self._rq_l_util = self._line_utilization(sub_id=sub_id)
        self._rq_t_util = self._transformer_utilization(sub_id=sub_id)
        utilization = pd.concat([self._rq_l_util, self._rq_t_util], axis=1).max(axis=1)

        prices = [price_func(u) for u in utilization.values]
        response = pd.Series(index=request.index, data=np.zeros(len(request)))
        response.loc[utilization.index] = prices
        response.replace(to_replace=0, method='ffill', inplace=True)

        return response

    def commit(self, request: pd.Series, node_id: str):
        sub_id = self.mapper[node_id]
        self.demand.loc[(self.demand.index.get_level_values(0) == node_id) &
                        (self.demand.index.get_level_values(1).isin(request.index)), 'power'] += request.values

        snapshots = self.grid.sub_networks[sub_id]['model'].snapshots
        self.line_utilization[sub_id].loc[snapshots, self._rq_l_util.columns] = self._rq_l_util.values
        self.line_utilization[sub_id].fillna(method='ffill', inplace=True)
        self.transformer_utilization[sub_id].loc[snapshots, 'utilization'] = self._rq_t_util.values.flatten()
        self.transformer_utilization[sub_id].fillna(method='ffill', inplace=True)

        # for sub_id in self.mapper.unique():
        #     snapshots = list(pd.DataFrame(self.fixed_power).drop_duplicates().index)
        #     print(len(snapshots))
        #     self.grid.sub_networks[sub_id].snapshots = snapshots
        #     for index in self.mapper[self.mapper == sub_id].index:
        #         demand = self.fixed_power[index].loc[snapshots] / 1000
        #         self.grid.sub_networks[sub_id].loads_t.p_set[f'{index}_consumer'] = demand.loc[snapshots]
        #     self.grid.run_power_flow(sub_id=sub_id)
        #     self.line_lock = self._line_utilization(sub_id)
        #     self.transformer_lock = self._transformer_utilization(sub_id)
        #     self.set_utilization()

    # def set_utilization(self):
    #     lines = deepcopy(self.line_lock)
    #     self.line_utilization.loc[lines.index, lines.columns] = lines
    #     transformer = deepcopy(self.transformer_lock)
    #     self.transformer_utilization.loc[transformer.index, transformer.columns] = transformer.values


    # def get_price(self, request: dict = None, d_time: datetime = None):
    #     # --> set current demand at each node
    #     demand = deepcopy(self.fixed_power)
    #     node_id = [key for key in request.keys()][0]
    #     max_duration = 0
    #     for parameters in request.values():
    #         for power, duration in parameters:
    #             demand[node_id][d_time:d_time + td(minutes=duration)] += power
    #             max_duration = max(duration, max_duration)
    #     # --> get unique timestamps
    #     snapshots = list(pd.DataFrame(demand).loc[d_time:d_time + td(minutes=max_duration)].drop_duplicates().index)
    #     # --> determine sub grid
    #     sub_id = self.mapper[node_id]
    #     self.grid.sub_networks[sub_id].snapshots = snapshots
    #     for index in self.mapper[self.mapper == sub_id].index:
    #         self.grid.sub_networks[sub_id].loads_t.p_set[f'{index}_consumer'] = demand[index].loc[snapshots] / 1000
    #     # --> run power flow calculation
    #     self.grid.run_power_flow(sub_id=sub_id)
    #     # --> get maximal utilization and calculate price
    #     self.line_lock = self._line_utilization(sub_id)
    #     self.transformer_lock = self._transformer_utilization(sub_id)
    #     max_ = max(self.line_lock.values.max(), self.transformer_lock.values.max())
    #     if max_ < 100:
    #         price = ((-np.log(1-np.power(max_/100, 1.5)) + 0.175) * 0.15) * 100
    #         return price, max_, sub_id
    #     return np.inf, 100, sub_id
    #
    # def get_results(self):
    #     line_utilization = self.line_utilization.replace(0, method='ffill')
    #     line = line_utilization.resample('15min').mean()
    #     lines = []
    #     for column in line_utilization.columns:
    #         if self.grid.model.lines.loc[column, 'bus0'] in self.grid.data['connected'].index \
    #                 or self.grid.model.lines.loc[column, 'bus1'] in self.grid.data['connected'].index:
    #             asset = 'outlet'
    #         else:
    #             asset = 'inlet'
    #
    #         lines += [dict(time=line.index,
    #                        iteration=[int(self.iteration)] * len(line),
    #                        scenario=[self.scenario] * len(line),
    #                        id_=[column] * len(line),
    #                        sub_id=[int(self.grid.model.lines.loc[column, 'sub_network'])] * len(line),
    #                        asset=[asset] * len(line),
    #                        utilization=line[column].values)]
    #
    #     transformer_utilization = self.transformer_utilization.replace(0, method='ffill')
    #     transformer = transformer_utilization.resample('15min').mean()
    #     transformers = []
    #     for column in transformer_utilization.columns:
    #         transformers += [dict(time=transformer.index,
    #                               iteration=[int(self.iteration)] * len(transformer),
    #                               scenario=[self.scenario] * len(transformer),
    #                               id_=[column] * len(transformer),
    #                               sub_id=[int(self.transformers_id[column])] * len(transformer),
    #                               asset=['transformer'] * len(transformer),
    #                               utilization=transformer[column].values)]
    #
    #     return lines, transformers


if __name__ == "__main__":
    cp = CapacityProvider(**dict(start_date='2022-02-01', end_date='2022-02-02', scenario=None, iteration=None))

