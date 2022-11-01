import logging
import multiprocessing as mp

import numpy as np
import overpass as osm
from shapely.geometry import Point, Polygon
from tqdm import tqdm


class GetNoInformation(Exception):
    pass


class GeoInformation:
    def __init__(
        self,
        api_endpoint: str = "https://maps.mail.ru/osm/tools/overpass/api/interpreter",
    ):
        self._geo_api = osm.API(endpoint=api_endpoint)
        self._logger = logging.getLogger("smartdso.geo")
        self.poi_s = []

        self._worker = mp.Pool(3)

    def _query_data(self, query):
        try:
            geojson = self._geo_api.get(query)
            return geojson
        except:
            return dict(features=[])

    def _get_nearest_feature(self, poi, features):
        features_coords = [feature["geometry"]["coordinates"] for feature in features]
        try:
            shapes = [
                Polygon([Point(tuple(coord)) for coord in feature])
                for feature in features_coords
            ]
            distances = np.asarray(
                [
                    poi.distance(shape.centroid) if not shape.is_empty else np.inf
                    for shape in shapes
                ]
            )
            index = np.argmin(distances)
            center = shapes[index].centroid
            return features[index], center
        except Exception as e:
            self._logger.error(repr(e))
            return None, None

    def _get_building_info(self, lat: float, lon: float):
        poi = Point(lon, lat)
        for distance in range(1, 11):
            self._logger.info(
                f"start searching for point lon: {lon}, lat: {lat} with distance: {distance}m"
            )
            geojson = self._query_data(
                f"way(around:{distance}, {lat}, {lon})[building=yes];out qt geom;"
            )
            if len(geojson["features"]) > 0:
                feature, center = self._get_nearest_feature(poi, geojson["features"])
                if center is not None:
                    self._logger.info(f"find feature for point lon: {lon}, lat: {lat}")
                    return feature, center
        return None, None

    def get_information(self):
        information = []
        for poi in tqdm(self.poi_s):
            information += [self._get_building_info(lon=poi[0], lat=poi[1])]
        return information


if __name__ == "__main__":
    geo_coder = GeoInformation()
    geo_coder.poi_s = [(6.48609, 50.80257) for _ in range(5)]
    x = geo_coder.get_information()
    # f, c = geo_coder.get_building_info(lat=50.80257, lon=6.48609)
