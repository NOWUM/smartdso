import logging
import pandas as pd
import numpy as np
from copy import deepcopy
from datetime import timedelta as td, datetime
from tqdm import tqdm

from gridLib.model import GridModel
logging.getLogger('pypsa').setLevel('ERROR')


class CapacityProvider:

    def __init__(self, *args, **kwargs):
        # --> build grid model and set simulation horizon
        self.grid = GridModel()
        self.mapper = self.grid.model.buses['sub_network']
        self.mapper = self.mapper[self.mapper.index.isin(self.grid.data['connected'].index)]
        # --> dictionary to store power values for each node
        time_range = pd.date_range(start=pd.to_datetime(kwargs['start_date']),
                                   end=pd.to_datetime(kwargs['end_date']) + td(days=1), freq='min')[:-1]
        self.fixed_power = {i: pd.Series(index=time_range, data=np.zeros(len(time_range)), name='power')
                            for i in self.grid.data['connected'].index}

        self.max_transformer_capacity = dict()
        for sub_id in self.mapper.unique():
            transformers = self.grid.data['transformers']
            buses_in_sub = self.grid.sub_networks[sub_id].buses.index
            s_max = transformers.loc[transformers['bus0'].isin(buses_in_sub), 's_nom'].values[0]
            self.max_transformer_capacity[sub_id] = s_max

        data = {i: np.zeros(len(time_range)) for i in self.grid.model.lines.index}
        self.line_utilization = pd.DataFrame(data=data, index=time_range)
        self.line_lock = None

        data = {i.replace('_slack', ''): np.zeros(len(time_range)) for i in self.grid.model.generators.index}
        self.transformer_utilization = pd.DataFrame(data=data, index=time_range)
        self.transformer_lock = None

    def set_fixed_power(self, data: pd.DataFrame):
        for index, series in self.fixed_power.items():
            power = data.loc[data['node_id'] == index].reset_index()
            power = power.groupby('t').sum()
            self.fixed_power[index][power.index] = power.values.flatten()
            self.fixed_power[index] = self.fixed_power[index].replace(0, method='ffill')

        for sub_id in tqdm(self.mapper.unique()):
            snapshots = list(pd.DataFrame(self.fixed_power).drop_duplicates().index)
            self.grid.sub_networks[sub_id].snapshots = snapshots
            for index in self.mapper[self.mapper == sub_id].index:
                demand = self.fixed_power[index].loc[snapshots] / 1000
                self.grid.sub_networks[sub_id].loads_t.p_set[f'{index}_consumer'] = demand.loc[snapshots]
            self.grid.run_power_flow(sub_id=sub_id)
            self.line_lock = self._line_utilization(sub_id)
            self.transformer_lock = self._transformer_utilization(sub_id)
            self.set_utilization()

    def set_utilization(self):
        lines = deepcopy(self.line_lock)
        self.line_utilization.loc[lines.index, lines.columns] = lines
        transformer = deepcopy(self.transformer_lock)
        self.transformer_utilization.loc[transformer.index, transformer.columns] = transformer.values

    def _line_utilization(self, sub_id: str):
        lines = self.grid.sub_networks[sub_id].lines_t.p0
        s_max = self.grid.sub_networks[sub_id].lines.loc[:, 's_nom']
        for column in lines.columns:
            lines[column] = np.abs(lines[column]) / s_max.loc[column] * 100
        return lines

    def _transformer_utilization(self, sub_id: str):
        transformer = self.grid.sub_networks[sub_id].generators_t.p
        transformer.columns = [column.replace('_slack', '') for column in transformer.columns]
        transformer = transformer/self.max_transformer_capacity[sub_id] * 100
        return transformer

    def get_price(self, request: dict = None, d_time: datetime = None):
        # --> set current demand at each node
        demand = deepcopy(self.fixed_power)
        node_id = [key for key in request.keys()][0]
        max_duration = 0
        for parameters in request.values():
            for power, duration in parameters:
                demand[node_id][d_time:d_time + td(minutes=duration)] += power
                max_duration = max(duration, max_duration)
        # --> get unique timestamps
        snapshots = list(pd.DataFrame(demand).loc[d_time:d_time + td(minutes=max_duration)].drop_duplicates().index)
        # --> determine sub grid
        sub_id = self.mapper[node_id]
        self.grid.sub_networks[sub_id].snapshots = snapshots
        for index in self.mapper[self.mapper == sub_id].index:
            self.grid.sub_networks[sub_id].loads_t.p_set[f'{index}_consumer'] = demand[index].loc[snapshots] / 1000
        # --> run power flow calculation
        self.grid.run_power_flow(sub_id=sub_id)
        # --> get maximal utilization and calculate price
        self.line_lock = self._line_utilization(sub_id)
        self.transformer_lock = self._transformer_utilization(sub_id)
        max_ = max(self.line_lock.values.max(), self.transformer_lock.values.max())
        if max_ < 100:
            price = ((-np.log(1-np.power(max_/100, 1.5)) + 0.175) * 0.15) * 100
            return price, max_, sub_id
        return np.inf, 100, sub_id

    def get_results(self):
        line_utilization = self.line_utilization.replace(0, method='ffill')
        line_utilization = line_utilization.resample('15min').mean()
        transformer_utilization = self.transformer_utilization.replace(0, method='ffill')
        transformer_utilization = transformer_utilization.resample('15min').mean()

        return line_utilization, transformer_utilization


if __name__ == "__main__":
    cp = CapacityProvider(**dict(start_date='2022-02-01', end_date='2022-02-02'))

