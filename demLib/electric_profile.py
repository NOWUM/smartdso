import numpy as np
import pandas as pd
from sqlalchemy import create_engine
import logging

from demLib.utils import get_holidays

DATABASE_URI = 'postgresql://opendata:opendata@10.13.10.41:5432/londondatastore'
engine = create_engine(DATABASE_URI)
path = r'./demLib/data/load_profiles/'

profiles = {
    'household': np.load(open(fr'{path}household.pkl', 'rb')),
    'business': np.load(open(fr'{path}business.pkl', 'rb')),
    'industry': np.load(open(fr'{path}industry.pkl', 'rb')),
    'agriculture':  np.load(open(fr'{path}agriculture.pkl', 'rb')),
    # 'light': pd.read_csv(fr'{path}light.csv')
}

winter = np.load(open(fr'{path}winter.pkl', 'rb'))
summer = np.load(open(fr'{path}summer.pkl', 'rb'))

logger = logging.getLogger('StandardLoadProfile-Generator')

class StandardLoadProfile:

    def __init__(self,
                 demandP: float,
                 type: str = 'household',
                 resolution: int = 96,
                 london_data: bool = False,
                 l_id: str = None):

        self.type = type                    # -> profile typ ['household', 'business', ...]
        self.demandP = demandP              # -> yearly energy demand
        self.profile = profiles[type]       # -> load profile data
        self.resolution = resolution        # -> set resolution
        self.london_data = london_data      # -> use smart meter data
        if london_data:
            self.data = self._get_london_data(l_id)

        self.winter = winter
        self.summer = summer

    def _get_london_data(self, l_id):
        try:
            query = f'SELECT "DateTime" as time, power from consumption where "LCLid" = \'{l_id}\' and' \
                    f'"DateTime" >= \'2013-01-01 00:00\' and "DateTime" < \'2014-01-01 00:00\''
            data = pd.read_sql(query, engine)
            data = data.set_index('time')
            consumption = data['power'].sum() * 0.5
            data['power'] /= consumption
            data['power'] *= self.demandP
            return data
        except Exception as e:
            logger.error(f'no smart meter data found - {repr(e)}')
            self.profile = profiles['household']
            logger.info('set london data to False')
            self.london_data = False

    def run_model(self, d: pd.Timestamp):

        doy, dow, year = d.dayofyear, d.dayofweek, d.year

        if self.london_data:
            try:
                demand = self.data.loc[self.data.index.dayofyear == doy]
                demand = np.asarray([[d, d] for d in demand['power'].values]).flatten()
                missing = max(96 - len(demand), 0)
                demand = np.hstack([demand, np.ones(missing) * np.mean(demand)])[:96]
            except Exception as e:
                logger.warning(f'no data found {repr(e)}')
                demand = 0.2 * np.ones(96)

        else:
            demand, f = np.zeros(96), self.demandP / 1e6
            if self.type == 'household':
                f *= -0.000000000392 * doy ** 4 + 0.00000032 * doy ** 3 - 0.0000702 * doy ** 2 + 0.0021 * doy + 1.24
            if dow == 6 or doy in get_holidays(year):
                demand = self.profile[:, 4] * f if doy in self.summer else self.profile[:, 1] * f
            elif dow < 5:
                demand = self.profile[:, 5] * f if doy in self.summer else self.profile[:, 2] * f
            elif dow == 5:
                demand = self.profile[:, 3] * f if doy in self.summer else self.profile[:, 0] * f
        if self.resolution == 60:
            return np.asarray([np.mean(demand[i:i + 3]) for i in range(0, 96, 4)], np.float).flatten()
        elif self.resolution == 1440:
            return np.asarray([demand[int(i)] for i in np.arange(0, 96, 1/15)], np.float).flatten()
        else:
            return demand
