import secrets
from tqdm import tqdm
import numpy as np
import pandas as pd
from geopy.extra.rate_limiter import RateLimiter
from geopy.geocoders import Nominatim
from shapely.geometry import Point
from shapely.wkt import loads

tqdm.pandas()


class LonLatFinder:

    def __init__(self, consumers: pd.DataFrame):

        geo_locator = Nominatim(user_agent=secrets.token_urlsafe(8))
        self.geocode = RateLimiter(geo_locator.geocode, min_delay_seconds=1.2)

        def convert_address(a: str):
            a_parts = a.split(" ")
            a = ""
            for part in a_parts:
                try:
                    _ = float(part)
                    a += " " + part
                    break
                except Exception as e:
                    a += " " + part
            # a = a_parts[0] + ' ' + a_parts[1]
            adr_str = f"{a} 52525 Heinsberg"
            return adr_str

        used_profiles = ["H01", "G01", "G11", "G21", "G31", "G41", "G51", "G61"]
        self.consumers = consumers.loc[consumers['profile'].isin(used_profiles)]
        self.consumers = self.consumers.reset_index()
        self.consumers['address'] = self.consumers['id_'].apply(convert_address)

    def find_coords(self):
        print('searching for address data')
        self.consumers["location"] = self.consumers["address"].progress_apply(self.geocode)
        self.consumers["lat"] = self.consumers["location"].apply(lambda x: x.latitude if x else None)
        self.consumers["lon"] = self.consumers["location"].apply(lambda x: x.longitude if x else None)
        self.consumers["shape"] = [Point(row.lon, row.lat) for _, row in self.consumers.iterrows()]

    def map_to_grid(self, grid_consumers: pd.DataFrame):

        def map_profile(pr):
            if "H" in pr:
                return "H0"
            if "G" in pr:
                return "G0"

        grid_consumers["jeb"] = 0

        consumers = self.consumers.set_index(['id_'])

        for consumer_id, consumer in grid_consumers.iterrows():
            min_distance, idx = np.inf, None
            for index, row in consumers.iterrows():
                distance = consumer["shape"].distance(row["shape"])
                if distance < min_distance:
                    min_distance = distance
                    idx = index

            grid_consumers.at[consumer_id, 'jeb'] = consumers.loc[idx, 'jeb']
            grid_consumers.at[consumer_id, 'profile'] = map_profile(consumers.loc[idx, 'profile'])

        return grid_consumers
