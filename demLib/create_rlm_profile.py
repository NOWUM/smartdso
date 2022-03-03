import numpy as np
from demandlib.electric_profile import StandardLoadProfile, get_holidays

from sqlalchemy.engine import create_engine
import pandas as pd
import matplotlib.pyplot as plt

winter = np.asarray(np.load(open(r'./demandlib/data/winter.pkl', 'rb')))
summer = np.asarray(np.load(open(r'./demandlib/data/summer.pkl', 'rb')))
database_oep = create_engine(f'postgresql://opendata:opendata@10.13.10.41:5432/oep', connect_args={"application_name": 'rlm'})


bdew_profiles = {
    'household': np.asarray(np.load(open(r'./demandlib/data/household.pkl', 'rb'))),
    'business': np.asarray(np.load(open(r'./demandlib/data/business.pkl', 'rb'))),
    'industry': np.asarray(np.load(open(r'./demandlib/data/industry.pkl', 'rb'))),
    'agriculture':  np.asarray(np.load(open(r'./demandlib/data/agriculture.pkl', 'rb')))
}

def get_power_households(demand):
    profile = StandardLoadProfile(demandP=demand, type='household', hourly=False)
    power = [profile.run_model(date) for date in pd.date_range(start='2019-01-01', freq='D', periods=365)]
    return np.asarray(power).reshape((-1,))


def get_power_business(demand):
    profile = StandardLoadProfile(demandP=demand, type='business', hourly=False)
    power = [profile.run_model(date) for date in pd.date_range(start='2019-01-01', freq='D', periods=365)]
    return np.asarray(power).reshape((-1,))


def get_power_agriculture(demand):
    profile = StandardLoadProfile(demandP=demand, type='agriculture', hourly=False)
    power = [profile.run_model(date) for date in pd.date_range(start='2019-01-01', freq='D', periods=365)]
    return np.asarray(power).reshape((-1,))


def get_power_oep():
    query = f'select sum(sector_consumption_residential) as household, sum(sector_consumption_retail) as business,' \
            f'sum(sector_consumption_industrial) as industry, sum(sector_consumption_agricultural) as agriculture ' \
            f'from demand where version=\'v0.4.5\''

    oep_data = pd.read_sql(query, database_oep)
    return oep_data.to_dict(orient='index')[0]


def get_power_entsoe():
    df = pd.read_csv(r'./demandlib/data/total_load_germany.csv', index_col=0)
    df.index = pd.date_range(start='2019-01-01', freq='15min', periods=len(df))
    df.index.name = 'time'
    del df[df.columns[0]]
    df.columns = ['total demand']
    return df['total demand'].to_numpy()


if __name__ == "__main__":
    energy_oep = get_power_oep()
    energy_household = energy_oep['household'] * 10**6
    power_ts_household = get_power_households(energy_household)
    power_ts_entsoe = get_power_entsoe() * 10**3

    power_ts_rlm = power_ts_entsoe - power_ts_household

    power = pd.DataFrame(data=dict(value=power_ts_rlm),
                         index=pd.date_range(start='2019-01-01', freq='15min', periods=len(power_ts_rlm)))

    power['value'] = power['value']/(sum(power['value'])/4) * 1000 * 1000

    profiles = {'summer': {'sa': [],
                           'so': [],
                           'wt': []},
                'winter': {'sa': [],
                           'so': [],
                           'wt': []},
                'rest':   {'sa': [],
                           'so': [],
                           'wt': []}}

    for d in pd.date_range(start='2019-01-01', freq='D', periods=365):
        doy = d.dayofyear
        dow = d.dayofweek

        if doy in summer:
            if dow == 6 or doy in get_holidays(doy):
                profiles['summer']['so'].append(power[power.index.dayofyear == doy].to_numpy())
            elif dow < 5:
                profiles['summer']['wt'].append(power[power.index.dayofyear == doy].to_numpy())
            elif dow == 5:
                profiles['summer']['sa'].append(power[power.index.dayofyear == doy].to_numpy())
        elif doy in winter:
            if dow == 6 or doy in get_holidays(doy):
                profiles['winter']['so'].append(power[power.index.dayofyear == doy].to_numpy())
            elif dow < 5:
                profiles['winter']['wt'].append(power[power.index.dayofyear == doy].to_numpy())
            elif dow == 5:
                profiles['winter']['sa'].append(power[power.index.dayofyear == doy].to_numpy())
        else:
            if dow == 6 or doy in get_holidays(doy):
                profiles['rest']['so'].append(power[power.index.dayofyear == doy].to_numpy())
            elif dow < 5:
                profiles['rest']['wt'].append(power[power.index.dayofyear == doy].to_numpy())
            elif dow == 5:
                profiles['rest']['sa'].append(power[power.index.dayofyear == doy].to_numpy())

    new_profiles = []
    for period in ['winter', 'summer', 'rest']:
        print(period)
        for day, data in profiles[period].items():
            x = np.asarray(data).reshape((-1, 96))
            profile = np.mean(x, axis=0)
            new_profiles.append(profile)

    x = np.asarray(new_profiles)
    new_profiles = np.asarray(new_profiles).T

    np.save('industry', new_profiles)
