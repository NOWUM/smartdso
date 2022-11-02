from datetime import date, timedelta
import numpy as np
import pandas as pd
import scipy.stats as sps

FLOW_RATES = {'shower': sps.norm(loc=8, scale=2),
              'bath': sps.norm(loc=14, scale=2),
              'small': sps.norm(loc=1, scale=2),
              'medium': sps.norm(loc=6, scale=0.7)}


WATER_PORTION = {'shower': 0.14,
                 'bath': 0.10,
                 'small': sps.norm(loc=1, scale=2),
                 'medium': sps.norm(loc=6, scale=0.7)}


def build_water_probability():
    r = {}
    index = pd.date_range(start='2022-01-01', freq='min', periods=1440)
    # -> shower probability
    prob = {}
    peak_1 = sps.norm(loc=6, scale=0.7)
    peak_2 = sps.norm(loc=19, scale=0.7)
    for t in index:
        hour = t.hour + t.minute/60
        if 4.8 <= hour <= 7.2:
            prob[t] = (peak_1.pdf(hour) / peak_1.pdf(6)) * 0.25
        elif 7.2 < hour <= 18:
            prob[t] = 0.014
        elif 17 <= hour <= 21:
            prob[t] = max((peak_2.pdf(hour) / peak_2.pdf(19)) * 0.09, 0.014)
        elif 21 < hour <= 23:
            prob[t] = 0.014
        else:
            prob[t] = 0
    shower = np.asarray([*prob.values()])
    shower /= shower.sum()
    r['shower'] = pd.Series(data=shower.cumsum(), index=index)

    # -> small and medium
    prob = {}
    for t in index:
        hour = t.hour + t.minute / 60
        if 4.8 <= hour <= 23:
            prob[t] = 0.055
        else:
            prob[t] = 0
    sm = np.asarray([*prob.values()])
    sm /= sm.sum()
    r['small'] = pd.Series(data=sm.cumsum(), index=index)
    r['medium'] = pd.Series(data=sm.cumsum(), index=index)

    # - bath
    prob = {}
    peak_1 = sps.norm(loc=15, scale=3)
    peak_2 = sps.norm(loc=18, scale=0.7)
    for t in index:
        hour = t.hour + t.minute / 60
        if 7 <= hour <= 17:
            prob[t] = (peak_1.pdf(hour) / peak_1.pdf(15)) * 0.06
        elif 17 < hour:
            demand_1 = max((peak_2.pdf(hour) / peak_2.pdf(18)), 0) * 0.22
            demand_2 = max((peak_1.pdf(hour) / peak_1.pdf(15)), 0) * 0.06
            prob[t] = max(demand_1, demand_2)
        else:
            prob[t] = 0

    bath = np.asarray([*prob.values()])
    bath /= bath.sum()
    r['bath'] = pd.Series(data=bath.cumsum(), index=index)

    return r


def get_holidays(year):
    # -- get eastern
    a = year % 19
    b = year // 100
    c = year % 100
    d = (19 * a + b - b // 4 - ((b - (b + 8) // 25 + 1) // 3) + 15) % 30
    e = (32 + 2 * (b % 4) + 2 * (c // 4) - d - (c % 4)) % 7
    f = d + e - 7 * ((a + 11 * d + 22 * e) // 451) + 114
    month = f // 31
    day = f % 31 + 1
    easter = date(year, month, day)

    # -- holidays in germany
    holidays = []
    holidays.append(easter)
    holidays.append(easter - timedelta(days=2))                 # -- Karfreitag
    holidays.append(easter + timedelta(days=1))                 # -- Ostermontag
    holidays.append(easter + timedelta(days=39))                # -- Christihimmelfahrt
    holidays.append(easter + timedelta(days=49))                # -- Pfingstsonntag
    holidays.append(easter + timedelta(days=50))                # -- Pfingstmontag
    holidays.append(easter + timedelta(days=60))                # -- Fronleichnam
    holidays.append(date(year, 12, 24))                         # -- 1. Weihnachtstag
    holidays.append(date(year, 12, 25))                         # -- 1. Weihnachtstag
    holidays.append(date(year, 12, 26))                         # -- 2. Weihnachtstag
    holidays.append(date(year, 12, 31))                         # -- 2. Weihnachtstag
    holidays.append(date(year, 1, 1))                           # -- Neujahr
    holidays.append(date(year, 5, 1))                           # -- 1. Mai
    holidays.append(date(year, 10, 3))                          # -- Tag der deutschen Einheit
    holidays.append(date(year, 10, 31))                         # -- Reformationstag

    return np.asarray([h.timetuple().tm_yday for h in holidays])


if __name__ == "__main__":
    prob = build_water_probability()
    p = prob['shower'].values
    num_per_day = 2
    times = np.sort([np.argwhere(np.random.uniform() < p)[0][0] for _ in range(num_per_day)])
    flow_rate_shower = sps.norm(8, 2)
    volumes = [flow_rate_shower.ppf(np.random.uniform()) for _ in range(num_per_day)]
    duration = 5





