from collections import defaultdict

import geopandas as gpd
import numpy as np
import pandas as pd
import plotly.graph_objs as go
from shapely.geometry import Point, Polygon
from tqdm import tqdm

BUILDINGS = pd.read_csv(r"./gridLib/converter/buildings.csv", index_col=0)


class HeatConverter:

    def __init__(self, heat_path: str):
        self.heat_demand = gpd.read_file(heat_path)
        idx = [x is not None for x in self.heat_demand["GEBAEUDETY"].values]
        self.heat_demand = self.heat_demand.loc[idx]
        self.heat_demand.set_crs("epsg:25832", inplace=True)
        self.heat_demand.to_crs("epsg:4326", inplace=True)

    def add_heat_demand_to_consumers(self, consumers: pd.DataFrame):

        coords = consumers.loc[:, ["lon", "lat"]]
        coords = coords.drop_duplicates()
        min_lat, max_lat = coords["lat"].values.min(), coords["lat"].values.max()
        min_lon, max_lon = coords["lon"].values.min(), coords["lon"].values.max()

        p1 = [min_lon, max_lat]
        p2 = [max_lon, max_lat]
        p3 = [max_lon, min_lat]
        p4 = [min_lon, min_lat]

        bounding_box = Polygon([p1, p2, p3, p4])
        bounding_box = gpd.GeoSeries(bounding_box)
        bounding_box.set_crs("epsg:4326", inplace=True)
        bounding_box = gpd.GeoDataFrame(bounding_box, columns=["geometry"])
        self.heat_demand = gpd.overlay(self.heat_demand, bounding_box, how="intersection")

        demand = dict()

        for consumer_id, coord in tqdm(coords.iterrows(), total=coords.shape[0]):
            lon, lat = coord.lon, coord.lat
            contains = lambda x: x.contains(Point((lon, lat)))
            idx = self.heat_demand["geometry"].apply(contains)
            rows = self.heat_demand.loc[idx]
            if rows.empty:
                distance = lambda x: x.distance(Point((lon, lat)))
                distance = self.heat_demand["geometry"].apply(distance)
                idx = distance == distance.min()
                rows = self.heat_demand.loc[idx]

            building_type = rows["GEBAEUDETY"].values[0]
            building_years = BUILDINGS.loc[building_type]
            heat_demand = rows["spez_WB_HU"].values[0]
            building_years = building_years.loc[building_years > heat_demand]
            if building_years.empty:
                year = 1918
            else:
                year = float(building_years.index[-1])
            # includes 13.5 kWh/m²a warm water demand
            demand[consumer_id] = (rows["WB_HU"].values[0], year)

        consumers["demandQ"] = 0
        consumers["year"] = 1918

        for id_, values in demand.items():
            consumers.at[id_, "demandQ"] = values[0]
            consumers.at[id_, "year"] = values[1]

        return consumers['year'].values, consumers['demandQ'].values
