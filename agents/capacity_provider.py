import logging
import os
from datetime import datetime
from datetime import timedelta as td

import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from shapely.wkt import loads

from gridLib.model import GridModel

logging.getLogger("pypsa").setLevel("ERROR")
logger = logging.getLogger("smartdso.capacity_provider")


class CapacityProvider:
    def __init__(
        self,
        name: str,
        sim: int,
        start_date: datetime,
        end_date: datetime,
        database_uri: str,
        grid_data: str,
        steps: int = 96,
        resolution: str = '15min',
        write_grid_to_gis: bool = True,
        sub_grid: int = -1,
        *args,
        **kwargs,
    ):
        self.name = name
        self.sim = sim

        self.T = steps
        self.time_range = pd.date_range(start=start_date, end=end_date + td(days=1), freq=resolution)[:-1]
        self.date_range = pd.date_range(start=start_date, end=end_date, freq="d")
        self.resolution = resolution

        components = {}
        for component in ['nodes', 'transformers', 'lines', 'consumers']:
            c = pd.read_csv(fr"./gridLib/data/export/{grid_data}/{component}.csv", index_col=0)
            c["geometry"] = c["shape"].apply(loads)
            components[component] = c

        # -> build grid model and set simulation horizon
        self.grid = GridModel(**components)

        self.sub_grid = sub_grid
        if self.sub_grid != -1:
            self.sub_ids = [sub_grid]
        else:
            consumers = pd.read_csv(fr"./gridLib/data/export/{grid_data}/consumers.csv", index_col=0)
            sub_grids = consumers['sub_grid'].unique()
            self.sub_ids = list(sub_grids)

        self.line_utilization = {}
        self.transformer_utilization = {}
        for sub_id in self.sub_ids:
            line_names = self.grid.sub_networks[sub_id]["model"].lines.index
            index = self.time_range
            self.line_utilization[sub_id] = pd.DataFrame(columns=line_names, index=index)
            self.transformer_utilization[sub_id] = pd.DataFrame(columns=["utilization"], index=index)

        self.demand = pd.DataFrame()

        self._database = create_engine(database_uri)

        self._geo_info = dict(edges=[], nodes=[], transformers=[])

        for sub_id in self.sub_ids:
            for asset_type in self._geo_info.keys():
                df = self.grid.get_components(type_=asset_type, grid=sub_id)
                df.set_crs(crs="EPSG:4326", inplace=True)
                self._geo_info[asset_type] += [df]
                if write_grid_to_gis:
                    df["asset"] = asset_type
                    # df.set_index('name', inplace=True)
                    df.to_postgis(
                        name=f"{asset_type}_geo", con=self._database, if_exists="append"
                    )

        self.grid_fee = pd.Series(data=2.6 * np.ones(len(self.time_range)), index=self.time_range)

    def get_grid_fee(self, time_range: pd.DatetimeIndex = None):
        if time_range is None:
            time_range = self.time_range
        return self.grid_fee.loc[time_range]

    def get_sub_id_to_node_id(self, node_id: str):
        idx = self.grid.data['consumers']['bus0'] == node_id
        sub_id = self.grid.data['consumers'].loc[idx, 'sub_grid'].values[0]
        return sub_id

    def get_line_utilization(self, sub_id: int) -> pd.DataFrame:
        lines = self.grid.sub_networks[sub_id]["model"].lines_t.p0
        s_max = self.grid.sub_networks[sub_id]["model"].lines.loc[:, "s_nom"]
        for column in lines.columns:
            lines[column] = np.abs(lines[column]) / s_max.loc[column] * 100
        return lines

    def get_transformer_utilization(self, sub_id: int) -> pd.DataFrame:
        transformer = self.grid.sub_networks[sub_id]["model"].generators_t.p
        s_max = self.grid.sub_networks[sub_id]["s_max"]
        transformer = transformer / s_max * 100
        return transformer

    def get_price(self, util):
        if util >= 100:
            return 100
        else:
            price = ((-np.log(1 - np.power(util / 100, 1.5)) + 0.175) * 0.15) * 100
            return min(price, 100)

    def run_power_flow(self, data: pd.DataFrame, sub_id: int, end_of_day: bool = False, d_time: datetime = None):
        if end_of_day:
            steps = pd.date_range(start=d_time, periods=self.T, freq=self.resolution)
            snapshots = list(steps)
        else:
            demand_unique = data.drop_duplicates(subset=["node_id", "power"])
            snapshots = list(demand_unique["t"].unique())

        self.grid.sub_networks[sub_id]["model"].set_snapshots(snapshots)
        for node in data["node_id"].unique():
            # -> demand time series in [kW] to [MW]
            name = f"{node}_consumer"
            demand = data.loc[data["node_id"] == node, ["t", "power"]]
            demand["power"] /= 1000
            demand = demand.set_index("t")
            consumers_series = self.grid.sub_networks[sub_id]["model"].loads_t
            consumers_series['p_set'].loc[snapshots, name] = demand.loc[snapshots].values.flatten()
        self.grid.run_power_flow(sub_id=sub_id)

    def set_utilization(self, steps: pd.DatetimeIndex, sub_id: int):
        # -> get results
        # -> lines
        lu = self.get_line_utilization(sub_id=sub_id)
        l_names = lu.columns
        # -> transformers
        tu = self.get_transformer_utilization(sub_id=sub_id)
        t_names = ["utilization"]

        self.line_utilization[sub_id].loc[steps, l_names] = lu.values
        self.transformer_utilization[sub_id].loc[steps, t_names] = tu.values

    def set_fixed_power(self, data: pd.DataFrame) -> None:
        self.demand = data.groupby(["node_id", "t"]).sum()
        demand = self.demand.copy()
        for sub_id in self.sub_ids:
            # -> first select all nodes in sub grid
            idx_nodes = self.grid.data['consumers']['sub_grid'] == sub_id
            nodes_in_grid = self.grid.data['consumers'].loc[idx_nodes, 'bus0']
            # -> second select consumers in sub grid
            consumers_in_grid = demand.index.get_level_values("node_id").isin(nodes_in_grid)
            for date in self.date_range:
                steps = pd.date_range(start=date, periods=self.T, freq=self.resolution)
                demand_data = demand.loc[consumers_in_grid].reset_index()
                # -> run power flow calculation
                self.run_power_flow(data=demand_data, sub_id=sub_id, end_of_day=True, d_time=date)
                self.set_utilization(steps=steps, sub_id=sub_id)

    def handle_request(self, request: pd.Series = None, node_id: str = "") -> pd.Series:
        # -> get corresponding sub grid
        sub_id = self.get_sub_id_to_node_id(node_id)
        # -> set current demand at each node
        demand = self.demand.copy()
        # -> get demand for requested time range
        idx = demand.index.get_level_values(level="t").isin(request.index)
        demand = demand.loc[idx]
        # -> select corresponding node
        idx = demand.index.get_level_values(level="node_id") == node_id
        # -> add requested demand
        demand.loc[idx, 'power'] += request.values
        demand = demand.sort_index(level="t").reset_index()
        # -> run power flow calculation
        self.run_power_flow(data=demand, sub_id=sub_id)
        # ->  get results
        # -> lines
        lu = self.get_line_utilization(sub_id=sub_id).max(axis=1)
        lu_prev = self.line_utilization[sub_id].loc[request.index].max(axis=1)
        lu_max = pd.concat([lu, lu_prev], axis=1).fillna(0).max(axis=1)
        # -> transformers
        tu = self.get_transformer_utilization(sub_id=sub_id).max(axis=1)
        tu_prev = self.transformer_utilization[sub_id].loc[request.index].max(axis=1)
        tu_max = pd.concat([tu, tu_prev], axis=1).fillna(0).max(axis=1)
        # -> maximal grid utilization
        util_max = pd.concat([lu_max, tu_max], axis=1).fillna(0).max(axis=1)
        # -> calculate grid fee
        prices = [self.get_price(u) for u in util_max.values]
        response = pd.Series(index=request.index, data=prices)

        return response

    def set_demand(self, request: pd.Series, node_id: str) -> None:
        # -> get demand for requested time range
        idx_time = self.demand.index.get_level_values(level="t").isin(request.index)
        # -> select corresponding node
        idx_node = self.demand.index.get_level_values(level="node_id") == node_id
        self.demand.loc[idx_node & idx_time, "power"] += request.values
        # -> get sub grid
        sub_id = self.get_sub_id_to_node_id(node_id)
        # -> get steps
        steps = self.grid.sub_networks[sub_id]["model"].snapshots
        self.set_utilization(steps=steps, sub_id=sub_id)

    def _save_summary(self, d_time: datetime) -> None:

        time_range = pd.date_range(
            start=d_time, freq=RESOLUTION[self.T], periods=self.T
        )

        aggregate_functions = {
            "mean": lambda x: pd.DataFrame.mean(x, axis=1),
            "max": lambda x: pd.DataFrame.max(x, axis=1),
            "median": lambda x: pd.DataFrame.median(x, axis=1),
        }

        def build_data(
            d: pd.DataFrame, asset: str = "line", s_id: int = 0, tp: str = "mean"
        ):
            d.columns = ["value"]
            d["type"] = tp
            d["sub_id"] = s_id
            d["asset"] = asset
            d["scenario"] = self.scenario
            d["iteration"] = self.iteration
            d.index.name = "time"

            d = d.reset_index()
            d = d.set_index(
                ["time", "scenario", "iteration", "type", "asset", "sub_id"]
            )

            return d

        for sub_id in self.sub_ids:
            dataframe = self.line_utilization[sub_id]
            for key, function in aggregate_functions.items():
                data = pd.DataFrame(function(dataframe.loc[time_range, :]))
                data = build_data(data, asset="line", s_id=int(sub_id), tp=key)
                try:
                    data.to_sql(
                        "grid_summary",
                        self._database,
                        if_exists="append",
                        method="multi",
                    )
                except Exception as e:
                    logger.warning(f"server closed the connection {repr(e)}")

        for sub_id in self.sub_ids:
            dataframe = self.transformer_utilization[sub_id]
            for key, function in aggregate_functions.items():
                data = pd.DataFrame(function(dataframe.loc[time_range, :]))
                data = build_data(data, asset="transformer", s_id=int(sub_id), tp=key)

                try:
                    data.to_sql(
                        "grid_summary",
                        self._database,
                        if_exists="append",
                        method="multi",
                    )
                except Exception as e:
                    logger.warning(f"server closed the connection {repr(e)}")

    def _save_grid_asset(self, d_time: datetime) -> None:

        time_range = pd.date_range(start=d_time, freq=self.resolution, periods=self.T)

        for sub_id in self.sub_ids:
            dataframe = self.line_utilization[sub_id]
            for line in dataframe.columns:
                result = dataframe.loc[time_range, [line]]
                result.columns = ["utilization"]
                result["id_"] = line
                result["asset"] = "line"
                result["scenario"] = self.name
                result["iteration"] = self.sim
                result["sub_id"] = int(sub_id)
                result.index.name = "time"
                result = result.reset_index()
                result = result.set_index(["time", "scenario", "iteration", "id_"])
                try:
                    result.to_sql(
                        "grid_asset", self._database, if_exists="append", method="multi"
                    )
                except Exception as e:
                    logger.warning(f"server closed the connection {repr(e)}")

        for sub_id in self.sub_ids:
            result = self.transformer_utilization[sub_id].loc[
                time_range, ["utilization"]
            ]
            result["id_"] = self.grid.get_components(type_="transformers", grid=sub_id).name.values[0]
            result["asset"] = "transformer"
            result["scenario"] = self.name
            result["iteration"] = self.sim
            result["sub_id"] = int(sub_id)
            result.index.name = "time"
            result = result.reset_index()
            result = result.set_index(["time", "scenario", "iteration", "id_"])
            try:
                result.to_sql(
                    "grid_asset", self._database, if_exists="append", method="multi"
                )
            except Exception as e:
                logger.warning(f"server closed the connection {repr(e)}")

    def _run_end_of_day(self, d_time: datetime):
        for sub_id in self.sub_ids:
            self.run_power_flow(data=self.demand.reset_index(), sub_id=sub_id,
                                end_of_day=True, d_time=d_time)
            lu = self.get_line_utilization(sub_id=sub_id)
            tu = self.get_transformer_utilization(sub_id=sub_id)
            time_range = pd.date_range(
                start=d_time, periods=self.T, freq=self.resolution
            )
            self.line_utilization[sub_id].loc[
                time_range, lu.columns
            ] = lu.values
            self.transformer_utilization[sub_id].loc[
                time_range, "utilization"
            ] = tu.values.flatten()

    def save_results(self, d_time: datetime) -> None:
        self._run_end_of_day(d_time)
        self._save_grid_asset(d_time)
        #self._save_summary(d_time)
        #if self.iteration == 0:
        #    self._save_grid_asset(d_time)


if __name__ == "__main__":
    from config import SimulationConfig as Config

    config_dict = Config().get_config_dict()

    cp = CapacityProvider(**config_dict)
