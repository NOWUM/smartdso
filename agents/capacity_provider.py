import logging
import pandas as pd
import numpy as np
from copy import deepcopy
from datetime import timedelta as td, datetime

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
        len_ = len(time_range)
        self.utilization = {i: pd.Series(data=np.zeros(len_), index=time_range) for i in self.mapper.unique()}
        self.price = {i: pd.Series(data=np.zeros(len_), index=time_range) for i in self.mapper.unique()}
        self.congestion = {i: pd.Series(data=np.zeros(len_), index=time_range) for i in self.mapper.unique()}

    def set_fixed_power(self, data: pd.DataFrame):
        for index, series in self.fixed_power.items():
            x = data.loc[data['node_id'] == index]
            series.loc[x.index] = x['power'].to_numpy()
            series = series.replace(to_replace=0, method='ffill')
            self.fixed_power[index] = series
        self._set_base_utilization()

    def _set_base_utilization(self):
        demand = deepcopy(self.fixed_power)
        for sub_id in self.mapper.unique():
            snapshots = list(pd.DataFrame(demand).drop_duplicates().index)
            self.grid.sub_networks[sub_id].snapshots = snapshots
            for index in self.mapper[self.mapper == sub_id].index:
                self.grid.sub_networks[sub_id].loads_t.p_set[f'{index}_consumer'] = demand[index].loc[snapshots] / 1000
            self.grid.run_power_flow(sub_id=sub_id)
            utilization = self._get_line_utilization(sub_id)
            utilization['transformer'] = self._get_transformer_utilization(sub_id)
            self.utilization[sub_id].loc[utilization.index] = utilization.max(axis=1)

    def _get_line_utilization(self, sub_id: str):
        lines = self.grid.sub_networks[sub_id].lines_t.p0
        s_max = self.grid.sub_networks[sub_id].lines.loc[:, 's_nom']
        for column in lines.columns:
            lines[column] = np.abs(lines[column]) / s_max.loc[column] * 100
        return lines

    def _get_transformer_utilization(self, sub_id: str):
        transformers = self.grid.data['transformers']
        buses_in_sub = self.grid.sub_networks[sub_id].buses.index
        s_max = transformers.loc[transformers['bus0'].isin(buses_in_sub), 's_nom'].values[0]
        transformer = self.grid.sub_networks[sub_id].generators_t.p.values
        u_tf = np.abs(transformer/s_max) * 100
        return u_tf

    def get_price(self, request: dict = None, d_time: datetime = None):
        # --> set current demand at each node
        demand = deepcopy(self.fixed_power)
        node_id = [key for key in request.keys()][0]
        max_duration = 0
        for parameters in request.values():
            for power, duration in parameters:
                demand[node_id][d_time:d_time + td(minutes=duration)] += power
                if duration > max_duration:
                    max_duration = duration
        # --> get unique timestamps
        snapshots = list(pd.DataFrame(demand).loc[d_time:d_time + td(minutes=max_duration)].drop_duplicates().index)
        # --> determine sub grid
        sub_id = self.mapper[node_id]
        self.grid.sub_networks[sub_id].snapshots = snapshots
        for index in self.mapper[self.mapper == sub_id].index:
            self.grid.sub_networks[sub_id].loads_t.p_set[f'{index}_consumer'] = demand[index].loc[snapshots] / 1000
        # --> run power flow calculation
        self.grid.run_power_flow(sub_id=sub_id)
        # --> get maximal power and calculate price
        df = self._get_line_utilization(sub_id)
        u_tf = self._get_transformer_utilization(sub_id)
        maximal_utilization = max(max(df.max(axis=1).to_numpy()), u_tf.max())
        if maximal_utilization < 100:
            price = ((-np.log(1-np.power(maximal_utilization/100, 1.5)) + 0.175) * 0.15) * 100
            return price, maximal_utilization, sub_id
        else:
            self.congestion[sub_id][d_time] = 1
            return np.inf, 100, sub_id

    def set_charging(self, price: float, utilization: float, sub_id: str, d_time: datetime):
        self.price[sub_id][d_time] = price
        self.utilization[sub_id][d_time] = utilization


if __name__ == "__main__":
    cp = CapacityProvider(**dict(start_date='2022-02-01', end_date='2022-02-02'))

