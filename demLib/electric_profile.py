import logging
import pickle

import numpy as np
import pandas as pd
from sqlalchemy import create_engine

from demLib.utils import get_holidays

DATABASE_URI = "postgresql://opendata:opendata@10.13.10.41:5432/londondatastore"
DATA_PATH = r"./demLib/data/load_profiles/london_data.pkl"
LONDON_DATA = pickle.load(open(DATA_PATH, "rb"))

engine = create_engine(DATABASE_URI)
path = r"./demLib/data/load_profiles/"

profiles = {
    "household": np.load(rf"{path}household.pkl"),
    "business": np.load(rf"{path}business.pkl"),
    "industry": np.load(rf"{path}industry.pkl"),
    "agriculture": np.load(rf"{path}agriculture.pkl"),
    # 'light': pd.read_csv(fr'{path}light.csv')
}

winter = np.load(rf"{path}winter.pkl")
summer = np.load(rf"{path}summer.pkl")

logger = logging.getLogger("smartdso.slp_generator")


class StandardLoadProfile:
    def __init__(
        self,
        demandP: float,
        type: str = "household",
        resolution: int = 96,
        london_data: bool = False,
        use_data_base: bool = False,
        l_id: str = None,
    ):

        self.type = type  # -> profile typ ['household', 'business', ...]
        self.demandP = demandP  # -> yearly energy demand
        self.profile = profiles[type]  # -> load profile data
        self.resolution = resolution  # -> set resolution
        self.london_data = london_data  # -> use smart meter data

        if london_data:
            self.data = self._get_london_data(l_id)

        self.winter = winter
        self.summer = summer

    def _get_london_data(self, l_id):
        try:
            data = LONDON_DATA[l_id].copy()
            data["power"] *= self.demandP
            return data

        except Exception as e:
            logger.error(f"no smart meter data found - {repr(e)}")
            self.profile = profiles["household"]
            logger.info("set london data to False")
            self.london_data = False

    def run_model(self, d: pd.Timestamp):

        doy, dow, year = d.dayofyear, d.dayofweek, d.year

        if self.london_data:
            try:
                demand = self.data.loc[self.data.index.dayofyear == doy]
                demand = np.asarray([[d, d] for d in demand["power"].values]).flatten()
                missing = max(self.resolution - len(demand), 0)
                demand = np.hstack([demand, np.ones(missing) * np.mean(demand)])[:self.resolution]
            except Exception as e:
                logger.warning(f"no data found {repr(e)}")
                demand = 0.2 * np.ones(self.resolution)

        else:
            demand, f = np.zeros(self.resolution), self.demandP / 1e6
            if self.type == "household":
                f *= (
                    -0.000000000392 * doy**4
                    + 0.00000032 * doy**3
                    - 0.0000702 * doy**2
                    + 0.0021 * doy
                    + 1.24
                )
            if dow == 6 or doy in get_holidays(year):
                demand = (
                    self.profile[:, 4] * f
                    if doy in self.summer
                    else self.profile[:, 1] * f
                )
            elif dow < 5:
                demand = (
                    self.profile[:, 5] * f
                    if doy in self.summer
                    else self.profile[:, 2] * f
                )
            elif dow == 5:
                demand = (
                    self.profile[:, 3] * f
                    if doy in self.summer
                    else self.profile[:, 0] * f
                )
        if self.resolution == 60:
            return np.asarray(
                [np.mean(demand[i : i + 3]) for i in range(0, self.resolution, 4)], np.float
            ).flatten()
        elif self.resolution == 1440:
            return np.asarray(
                [demand[int(i)] for i in np.arange(0, self.resolution, 1 / 15)], np.float
            ).flatten()
        else:
            return demand


if __name__ == "__main__":
    from matplotlib import pyplot as plt

    s = StandardLoadProfile(demandP=6000)
    x_s = []

    for date in pd.date_range(start='2018-01-01', periods=7, freq='d'):
        x = s.run_model(d=date)
        x_s.append(x)

    x_s = np.array(x_s).flatten()
    df = pd.DataFrame(data={'power': x_s}, index=pd.date_range(start='2018-01-01',
                                                               periods=len(x_s),
                                                               freq='15min'))
    df.to_excel('mein_slp.xlsx')

    s = StandardLoadProfile(demandP=6000, london_data=True, l_id='MAC005555')
    x_s = []

    for date in pd.date_range(start='2018-01-01', periods=7, freq='d'):
        x = s.run_model(d=date)
        x_s.append(x)

    x_s = np.array(x_s).flatten()
    df = pd.DataFrame(data={'power': x_s}, index=pd.date_range(start='2018-01-01',
                                                               periods=len(x_s),
                                                               freq='15min'))

    df.to_excel('mein_london_slp.xlsx')

