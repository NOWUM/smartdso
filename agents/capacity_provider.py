import logging
import pandas as pd
import numpy as np
from tqdm import tqdm
from gridLib.model import GridModel
import sqlite3
logging.getLogger('pypsa').setLevel('ERROR')


class CapacityProvider:

    def __init__(self, tqdm: bool = False):
        self.grid = GridModel()
        self._logger = logging.getLogger('CapacityProvider')
        self._logger.setLevel('INFO')
        self._database = sqlite3.connect('simulation.db')
        self._tqdm = not tqdm


    def _get_demand(self, t1=None, t2=None):

        if t1 is None or t2 is None:
            query = 'Select node_id, t, sum(power) as power from daily_demand group by node_id, t'
        else:
            query = f'Select node_id, t, sum(power) as power from daily_demand where ' \
                    f't >= {t1} and t <= {t2} group by node_id, t'

        df = pd.read_sql(query, self._database)
        node_data = {node: df.loc[df['node_id'] == node, 'power'].values for node in df['node_id'].unique()}
        node_data = pd.DataFrame(node_data)
        return node_data

    def _set_demand(self, demand, t1=None, t2=None):
        for node in tqdm(demand.columns, disable=self._tqdm):
            if t1 is None or t2 is None:
                self.grid.power_series[node] = demand[node].to_numpy().flatten()
            else:
                self.grid.power_series[node][t1:t2+1] = demand[node].to_numpy().flatten()

    def _get_result(self):
        lines = self.grid.model.lines_t.p0.replace(to_replace=0, method='ffill')
        s_max = self.grid.model.lines.loc[:, 's_nom']
        for column in lines.columns:
            lines[column] = np.abs(lines[column]) / s_max.loc[column] * 100
        return lines

    def plan_fixed_demand(self, daily_demand: pd.DataFrame = None):
        if daily_demand is None:
            daily_demand = self._get_demand()
            self._logger.info('get demand data from database')
        else:
            daily_demand = {node: daily_demand.loc[daily_demand['node_id'] == node, 'power'].values
                            for node in daily_demand['node_id'].unique()}
            daily_demand = pd.DataFrame(daily_demand)
            self._logger.info('get demand data from flexibility provider')
        self._set_demand(daily_demand)

        self._logger.info('run power flow calculation')
        self.grid.run_power_flow(fixed=True, snapshots=[])
        result = self._get_result()
        result.to_sql('lines', self._database, if_exists='replace')
        self._logger.info('finished power flow calculation')

        return result

    def get_price(self, request: pd.DataFrame = None):
        # ---> charging time range at fixed node
        t1 = request.index.get_level_values('t').min()
        t2 = request.index.get_level_values('t').max()
        node_id = request.index.get_level_values('node_id')[0]

        self._logger.info('collect data for current request')
        # ---> set current demand at each node
        demand = self._get_demand(t1=t1, t2=t2)
        demand = demand.round(2)
        self._set_demand(demand, t1, t2)
        # ---> add requested charging power
        demand[node_id] += request['power'].to_numpy()
        demand = demand.round(2)
        self._set_demand(demand, t1, t2)
        # ---> get unique timestamps
        snapshots = list(demand.drop_duplicates().index)
        self._logger.info('collection finished')

        # ---> run power flow calculation
        self.grid.run_power_flow(fixed=False, snapshots=snapshots)
        # ---> get maximal power and calculate price
        df = self._get_result()
        df = df.loc[(df.index.isin(snapshots))]
        maximal_utilization = max(df.max(axis=1).to_numpy())
        # print(maximal_utilization)
        if maximal_utilization < 100:
            prices = (-np.log(1-np.power(maximal_utilization/100, 1.5))+0.175)*100
            durations = np.asarray(list(np.diff(snapshots)) + [t2 - snapshots[-1] + 1])
            price = np.mean(durations * prices)
            return round(price/100, 2)
        else:
            return np.inf


if __name__ == "__main__":
    cp = CapacityProvider()


