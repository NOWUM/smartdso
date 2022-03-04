import os
import numpy as np
import names
from datetime import datetime, timedelta
from mobLib.mobility_demand import MobilityDemand
from carLib.car import Car

ev_ratio = int(os.getenv('EV_RATIO', 80))
ev_ratio = int(ev_ratio)/100

minimum_soc = os.getenv('MINIMUM_SOC', 30)

employee_ratio = os.getenv('EMPLOYEE_RATIO', 0.7)


def in_minutes(d_time):
    if isinstance(d_time, datetime):
        return d_time.hour * 60 + d_time.minute
    else:
        d_time = datetime.strptime(d_time, '%H:%M:%S') if len(d_time.split(':') == 3) \
            else datetime.strptime(d_time, '%H:%M')
        return d_time.hour * 60 + d_time.minute


class Resident:

    def __init__(self, mobility_types: list, type_: str = 'adult', last_name: str = 'nowum'):
        self.name = f'{names.get_first_name(gender=np.random.choice(["male", "female"], p=[0.5, 0.5]))} {last_name}'
        self.type = type_

        self.employee = True if 'work' in mobility_types else False

        self._mobility_generator = MobilityDemand(mobility_types)
        self.own_car = self._get_car_dependency()
        if self.own_car:
            self.car = Car(type=np.random.choice(a=['ev', 'fv'], p=[ev_ratio, 1-ev_ratio]))
        else:
            self.car = None

    def get_mobility_demand(self, date: datetime):
        dow = date.date().strftime("%A")
        if self.type == 'adult':
            return self._mobility_generator.mobility[dow]
        else:
            return []

    def _get_car_dependency(self):
        for demands in self._mobility_generator.mobility.values():
            for demand in demands:
                if demand['car_use']:
                    return True
        return False

    def get_mobility_days(self, type_='work'):
        if type_ == 'work':
            return self._mobility_generator.working_days
        elif type_ == 'errand':
            return self._mobility_generator.errand_days
        elif type_ == 'hobby':
            return self._mobility_generator.hobby_days
        else:
            return []

    def get_car_usage(self, date: datetime):
        mob = self.get_mobility_demand(date)           # get mobility demand for the current weekday
        usage = np.ones(1440)                          # charging opportunities equals 1
        consumption = np.zeros(1440)
        if self.own_car:
            for m in [x for x in mob if x['car_use']]:
                # determine departure and arrival times
                departure_time = datetime.strptime(m['start_time'], '%H:%M:%S') - timedelta(minutes=m['travel_time'])
                arrival_time = departure_time + timedelta(minutes=m['duration'] + 2 * m['travel_time'])
                # for the time_interval departure_time - arrival_time no charging opportunities are available --> 0
                usage[in_minutes(departure_time):in_minutes(arrival_time)] = 0

                mean_consumption = (m['distance'] * self.car.consumption / 100) / m['travel_time']
                to_journey = (in_minutes(departure_time), in_minutes(departure_time
                                                                     + timedelta(minutes=m['travel_time'])))
                back_home = (in_minutes(arrival_time - timedelta(minutes=m['travel_time'])), in_minutes(arrival_time))

                consumption[to_journey[0]:to_journey[1]] = mean_consumption
                consumption[back_home[0]:back_home[1]] = mean_consumption

            self.car.chargeable = usage
            self.car.driving = consumption

            return usage
        else:
            return np.zeros(1440)

    def plan_charging(self, d_time: datetime):
        index = in_minutes(d_time)
        chargeable = np.argwhere(self.car.chargeable == 1)
        chargeable = chargeable[chargeable >= index]

        car_in_use = np.argwhere(self.car.chargeable == 0)
        car_in_use = car_in_use[car_in_use >= index]

        if index in chargeable:
            t1 = index
            t2 = car_in_use[0] if len(car_in_use > 0) else 1440

            if self.car.soc < minimum_soc:
                total_energy = self.car.capacity - (self.car.capacity * self.car.soc)/100
                duration = total_energy/self.car.maximal_charging_power * 60
                if duration < (t2-t1):
                    return t1, t1 + int(duration)
                else:
                    return t1, t2
        return 0, 0

