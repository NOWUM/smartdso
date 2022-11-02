from collections import defaultdict
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point
from tqdm import tqdm


class PVConverter:

    def __init__(self, pv_path: str):
        self.potential = gpd.read_file(pv_path)
        self.potential.set_crs("epsg:25832", inplace=True)
        self.potential.to_crs("epsg:4326", inplace=True)

    def add_pv_to_consumers(self, consumers: pd.DataFrame):

        consumers["pv"] = None
        consumers = consumers.loc[consumers["demand_power"] > 0]
        consumer_nodes = consumers.loc[consumers["profile"] == "H0"]
        coords = consumer_nodes.loc[:, ["lon", "lat"]]

        buildings = defaultdict(list)
        for consumer_id, coord in tqdm(coords.iterrows(), total=coords.shape[0]):
            lon, lat = coord.lon, coord.lat

            distance = lambda x: x.distance(Point((lon, lat)))
            self.potential["distance"] = self.potential["geometry"].apply(distance)

            idx = self.potential["distance"].min() == self.potential["distance"]
            building = self.potential.loc[idx]
            if not building.empty:
                for id_ in building["OBJECTID"].values:
                    buildings[id_].append(consumer_id)

        systems = defaultdict(list)
        for building_id, consumer_ids in buildings.items():
            idx = self.potential["OBJECTID"] == building_id
            pv_system = self.potential.loc[idx]
            power = pv_system["kw_17"].values[0]
            # peak power in kW with 17% wirkleistung
            surface_tilt = pv_system["neigung"].values[0]
            if pv_system["richtung"].values[0] != -1:
                surface_azimuth = pv_system["richtung"].values[0]
            else:
                surface_azimuth = 180

            total_demand = consumers.loc[consumer_ids, "demand_power"].sum()

            for consumer_id in consumer_ids:
                demand = consumers.loc[consumer_ids, "demand_power"].values[0]
                pdc0 = (power / total_demand) * demand
                system = {"pdc0": round(pdc0, 1),
                          "surface_tilt": surface_tilt,
                          "surface_azimuth": surface_azimuth}
                systems[consumer_id].append(system)

            for consumer_id, systems in systems.items():
                consumers.at[consumer_id, "pv"] = str(systems)

        return consumers["pv"].values
