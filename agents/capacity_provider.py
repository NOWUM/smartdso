import logging
import os
import pandas as pd
import numpy as np
from copy import deepcopy
from datetime import timedelta as td, datetime

from gridLib.model import GridModel
import sqlite3
logging.getLogger('pypsa').setLevel('ERROR')

start_date = pd.to_datetime(os.getenv('START_DATE', '2022-02-01'))
end_date = pd.to_datetime(os.getenv('END_DATE', '2022-02-10'))
time_range = pd.date_range(start=start_date, end=end_date + td(days=1), freq='min')[:-1]


class CapacityProvider:

    def __init__(self):
        # ---> build grid model and set simulation horizon
        self.grid = GridModel()
        self.mapper = self.grid.model.buses['sub_network']
        self.mapper = self.mapper[self.mapper.index.isin(self.grid.data['connected'].index)]
        # ---> dictionary to store power values for each node
        self.fixed_power = {i: pd.Series(index=time_range, data=np.zeros(len(time_range)), name='power')
                            for i in self.grid.data['connected'].index}

        # ---> set logger
        self._logger = logging.getLogger('CapacityProvider')
        self._logger.setLevel('INFO')

    def set_fixed_power(self, data: pd.DataFrame):
        for index, series in self.fixed_power.items():
            x = data.loc[data['node_id'] == index]
            series.loc[x.index] = x['power'].to_numpy()
            series = series.replace(to_replace=0, method='ffill')
            self.fixed_power[index] = series

    def _get_result(self, sub_id):
        lines = self.grid.sub_networks[sub_id].lines_t.p0.replace(to_replace=0, method='ffill')
        s_max = self.grid.sub_networks[sub_id].lines.loc[:, 's_nom']
        for column in lines.columns:
            lines[column] = np.abs(lines[column]) / s_max.loc[column] * 100
        return lines

    def get_price(self, request: dict = None, d_time: datetime = None):
        self._logger.info('collect data for current request')
        # ---> set current demand at each node
        demand = deepcopy(self.fixed_power)
        node_id = [key for key in request.keys()][0]
        max_duration = 0
        for parameters in request.values():
            for power, duration in parameters:
                demand[node_id][d_time:d_time + td(minutes=duration)] += power
                if duration > max_duration:
                    max_duration = duration
        # ---> get unique timestamps
        snapshots = list(pd.DataFrame(demand).loc[d_time:d_time + td(minutes=max_duration)].drop_duplicates().index)
        # ---> determine sub grid
        sub_id = self.mapper[node_id]
        self._logger.info('collection finished')
        self.grid.sub_networks[sub_id].snapshots = snapshots
        for index in self.mapper[self.mapper == sub_id].index:
            self.grid.sub_networks[sub_id].loads_t.p_set[f'{index}_consumer'] = demand[index].loc[snapshots] / 1000
        # ---> run power flow calculation
        self.grid.run_power_flow(sub_id=sub_id)
        # ---> get maximal power and calculate price
        df = self._get_result(sub_id)
        # TODO: Check max values --> current only one value is used
        maximal_utilization = max(df.max(axis=1).to_numpy())
        if maximal_utilization < 100:
            return ((-np.log(1-np.power(maximal_utilization/100, 1.5)) + 0.175) * 0.15) * 100
        else:
            return np.inf


if __name__ == "__main__":
    cp = CapacityProvider()
    a = cp.grid.data['connected'].index[8]
    request = {a: [(630, 6), (500, 8)]}
    d_time = pd.to_datetime('2022-02-01')
    cp.get_price(request, d_time)
    cp.grid.plot()
    # cp.get_fixed_power()

