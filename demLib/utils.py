from datetime import date, timedelta
import numpy as np

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