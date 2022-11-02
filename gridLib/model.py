import logging
import os
import warnings

import geopandas as gpd
import pandas as pd
import pypsa
from shapely.wkt import loads

warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=pd.errors.PerformanceWarning)


class GridModel:
    def __init__(
            self,
            nodes: pd.DataFrame,
            lines: pd.DataFrame,
            transformers: pd.DataFrame,
            consumers: pd.DataFrame):

        logging.getLogger("pypsa").setLevel("ERROR")
        # --> set logger
        self._logger = logging.getLogger("smartdso.grid_model")
        self._logger.setLevel(logging.WARNING)

        self.model = pypsa.Network()  # -> total grid model
        self.sub_networks = {}  # -> dict for sub network models

        # -> build grid data
        grid_data = {
            "nodes": nodes,
            "transformers": transformers,
            "edges": lines,
            "consumers": consumers,
        }

        # -> used by a least one consumer
        idx = grid_data["nodes"].index.isin(grid_data["consumers"]["bus0"])
        grid_data["consumer_nodes"] = grid_data["nodes"].loc[idx]
        grid_data["voltage_ids"] = grid_data["consumer_nodes"]["voltage_id"].unique()

        self.data = grid_data.copy()

        # -> add busses to network --> each node is a bus
        idx = self.data["nodes"]["voltage_id"].isin(self.data["voltage_ids"])
        nodes = self.data["nodes"].loc[idx]
        self.model.madd(
            "Bus",
            names=nodes.index,
            v_nom=nodes.v_nom,
            x=nodes.lon,
            y=nodes.lat
        )

        # -> add edges to network to connect the busses
        idx = lines["bus0"].isin(nodes.index) | lines["bus1"].isin(nodes.index)
        edges = self.data["edges"]
        edges = edges.loc[idx]

        self.model.madd(
            "Line",
            names=edges.index,
            bus0=edges.bus0,
            bus1=edges.bus1,
            x=edges.x,
            r=edges.r,
            s_nom=edges.s_nom,
        )

        # -> add slack generators for each transformer with higher voltage
        idx = transformers["bus0"].isin(nodes.index) | transformers["bus1"].isin(nodes.index)
        transformers = self.data["transformers"]
        transformers = transformers.loc[idx]
        slack_nodes = transformers["bus0"].unique()
        for node in slack_nodes:
            self.model.add(
                "Generator",
                name=f"{node}_slack",
                bus=node,
                control="Slack"
            )

        # -> add consumers to node
        for node, consumer in self.data["consumer_nodes"].iterrows():
            self.model.add(
                "Load",
                name=f"{node}_consumer",
                bus=node)

        # -> determine sub networks and check consistency
        self.model.determine_network_topology()
        self.model.consistency_check()

        self._invalid_sub_grids = dict()

        for index in self.model.sub_networks.index:
            try:
                model = pypsa.Network()
                # -> add busses to network
                idx = self.model.buses["sub_network"] == index
                nodes = self.model.buses.loc[idx]
                model.madd(
                    "Bus",
                    names=nodes.index,
                    v_nom=nodes.v_nom,
                    x=nodes.x,
                    y=nodes.y
                )
                # -> add lines to network
                idx = self.model.lines["sub_network"] == index
                lines = self.model.lines.loc[idx]
                model.madd(
                    "Line",
                    names=lines.index,
                    bus0=lines.bus0,
                    bus1=lines.bus1,
                    x=lines.x,
                    r=lines.r,
                    s_nom=lines.s_nom,
                )
                # -> add slack generator to network
                generators = nodes["generator"].dropna()
                name = generators.values[0]
                generator = self.model.generators.loc[name]
                model.add(
                    "Generator",
                    name=name,
                    bus=generator.bus,
                    control="Slack"
                )
                # -> add consumers to network
                idx = self.model.loads["bus"].isin(nodes.index)
                consumers = self.model.loads.loc[idx]
                model.madd(
                    "Load",
                    names=consumers.index,
                    bus=consumers.bus
                )

                model.determine_network_topology()
                model.consistency_check()

                # -> get s nom from transformer
                idx = transformers["bus0"].isin(model.buses.index)
                transformer = transformers.loc[idx, "s_nom"]

                self.sub_networks[int(index)] = {
                    "model": model,
                    "s_max": transformer.values[0],
                }

            except Exception as e:
                self._logger.info(f"no valid sub grid " f"{repr(e)}")

    def get_components(
        self, type_: str = "edges", grid: str = "total"
    ) -> gpd.GeoDataFrame:
        if grid == "total":
            model = self.model
        else:
            model = self.sub_networks[str(grid)]["model"]

        if type_ == "nodes":
            data = self.data[type_].loc[self.data[type_].index.isin(model.buses.index)]
        elif type_ == "edges":
            data = self.data[type_].loc[self.data[type_].index.isin(model.lines.index)]
        else:
            i = self.data[type_]["bus0"].isin(model.buses.index) | self.data[type_][
                "bus1"
            ].isin(model.buses.index)
            data = self.data[type_].loc[i]
        data.index.name = "name"
        df = data.reset_index()
        return gpd.GeoDataFrame(df.loc[:, ["name", "geometry"]], geometry="geometry")

    def run_power_flow(self, sub_id: str):

        try:
            result_summary = self.sub_networks[sub_id]["model"].pf()
            converged = result_summary.converged
            for col in converged.columns:
                if any(converged[col]):
                    self._logger.warning(f"pfc for subgrid {col} not converged")
                else:
                    self._logger.info(f"pfc for subgrid {col} converged")

        except Exception as e:
            self._logger.error(repr(e))
            self._logger.error("error during power flow calculation")


if __name__ == "__main__":

    GRID_DATA = os.getenv("GRID_DATA", "dem")

    data_path = rf"./gridLib/data/export/{GRID_DATA}"
    # -> read nodes
    total_nodes = pd.read_csv(rf"{data_path}/nodes.csv", index_col=0)
    total_nodes["geometry"] = total_nodes["shape"].apply(loads)
    # -> read transformers
    total_transformers = pd.read_csv(rf"{data_path}/transformers.csv", index_col=0)
    total_transformers["geometry"] = total_transformers["shape"].apply(loads)
    # -> read edges
    total_edges = pd.read_csv(rf"{data_path}/edges.csv", index_col=0)
    total_edges["geometry"] = total_edges["shape"].apply(loads)
    # -> read known consumers
    # total_consumers = pd.read_csv(fr'{data_path}/grid_allocations.csv', index_col=0)
    total_consumers = pd.read_csv(rf"{data_path}/consumers.csv", index_col=0)

    model = GridModel(nodes=total_nodes, lines=total_edges, transformers=total_transformers,
                      consumers=total_consumers)
    b = total_consumers['bus0'].values[0]
    get_sub_grid = lambda x: int(model.model.buses.loc[x, 'sub_network'])
    total_consumers['sub_grid'] = total_consumers['bus0'].apply(get_sub_grid)

    # subs = [*model._invalid_sub_grids.values()]
    # edges = []
    # nodes = []
    # for sub in subs:
    #     e = total_edges.loc[sub.lines.index]
    #     e["shape"] = [wkt.loads(val) for val in e["shape"].values]
    #     busses = list(e["bus0"].values) + list(e["bus1"].values)
    #     n = total_nodes.loc[busses]
    #     edges.append(e)
    #     nodes.append(n)
    # edges = pd.concat(edges)
    # nodes = pd.concat(nodes)
    #
    # plt = get_plot(edges=edges, nodes=nodes)
    # plt.write_html("test.html")
