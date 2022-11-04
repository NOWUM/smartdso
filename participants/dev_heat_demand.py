from datetime import datetime, timedelta as td
from pvlib.pvsystem import PVSystem

import pandas as pd
import numpy as np

from demLib.heat_profile import StandardLoadProfile as HeatProfile
from demLib.electric_profile import StandardLoadProfile as PowerProfile
from participants.basic import BasicParticipant
from participants.resident import Resident
from agents.utils import WeatherGenerator

from pyomo.environ import (
    Binary,
    Expression,
    ConcreteModel,
    Constraint,
    ConstraintList,
    Objective,
    Piecewise,
    Reals,
    SolverFactory,
    Var,
    maximize,
    minimize,
    quicksum,
    value,
)


class HouseholdWithHeat(BasicParticipant):
    def __init__(
            self,
            residents: int,
            demand_power: float,
            demand_heat: float,
            database_uri: str,
            random: np.random.default_rng,
            london_data: bool = False,
            london_id: str = "MAC002957",
            ev_ratio: float = 0.5,
            pv_systems: list = None,
            grid_node: str = None,
            start_date: datetime = datetime(2022, 1, 1),
            end_date: datetime = datetime(2022, 1, 2),
            steps: int = 96,
            resolution: str = "15min",
            consumer_id: str = "nowum",
            strategy: str = "MaxPvCap",
            sub_grid: int = -1,
            weather: pd.DataFrame = None,
            tariff: pd.Series = None,
            grid_fee: pd.Series = None,
            *args,
            **kwargs,
    ):
        # -> initialize profile generator
        p_gen = PowerProfile(demandP=demand_power, london_data=london_data, l_id=london_id, resolution=steps)
        # -> initialize pv systems
        pv_systems = [PVSystem(module_parameters=system) for system in pv_systems]

        super().__init__(
            steps=steps,
            resolution=resolution,
            consumer_id=consumer_id,
            sub_grid=sub_grid,
            grid_node=grid_node,
            residents=residents,
            start_date=start_date,
            end_date=end_date,
            database_uri=database_uri,
            consumer_type="household",
            strategy=strategy,
            random=random,
            profile_generator=p_gen,
            pv_systems=pv_systems,
            weather=weather,
            tariff=tariff,
            grid_fee=grid_fee
        )

        self.heat_profile_gen = HeatProfile(demandQ=demand_heat)

        self._model = ConcreteModel()
        self._solver_type = "glpk"
        self._solver = SolverFactory(self._solver_type)

        self.volume_hw = (400 * 0.997 * 4.19 * (50-20)) / 3600
        self.volume_sh = (600 * 0.997 * 4.19 * 15) / 3600

    def get_heat_demand(self, d_time: pd.Timestamp):
        time_range = pd.date_range(start=d_time, periods=self.T, freq=self.resolution)
        temperature = self.weather.loc[time_range, 'temp_air'].values - 273.15
        r = self.heat_profile_gen.run_model(temperature=temperature)
        return r

    def get_power_demand(self, d_time: pd.Timestamp):
        time_range = pd.date_range(start=d_time, periods=self.T, freq=self.resolution)
        return self._data.loc[time_range, 'demand']

    def get_pv_generation(self, d_time: pd.Timestamp):
        time_range = pd.date_range(start=d_time, periods=self.T, freq=self.resolution)
        return self._data.loc[time_range, 'generation']

    def optimize(self, d_time: pd.Timestamp):
        demand_power = self.get_power_demand(d_time)

        heat_values = self.get_heat_demand(d_time)
        demand_hw = heat_values['hot_water']
        cop_hw = heat_values['cop_hot_water']

        demand_sh = heat_values['space_heating']
        cop_sh = heat_values['space_heating']

        generation = self.get_pv_generation(d_time)

        self._model.clear()

        self._model.power_wp = Var(self.t, within=Reals, bounds=(0, None))
        self._model.heat_wp_hw = Var(self.t, within=Reals, bounds=(0, 5))
        self._model.heat_wp_sh = Var(self.t, within=Reals, bounds=(0, 5))

        self._model.volume_hw = Var(self.t, within=Reals, bounds=(0, self.volume_hw))
        self._model.v_in_hw = Var(self.t,  within=Reals, bounds=(0, None))
        self._model.v_out_hw = Var(self.t, within=Reals, bounds=(0, None))

        self._model.volume_sh = Var(self.t, within=Reals, bounds=(0, self.volume_sh))
        self._model.v_in_sh = Var(self.t,  within=Reals, bounds=(0, None))
        self._model.v_out_sh = Var(self.t, within=Reals, bounds=(0, None))

        self._model.power_grid = Var(self.t, within=Reals, bounds=(None, None))
        self._model.power_grid_in = Var(self.t, within=Reals, bounds=(0, None))
        self._model.power_grid_out = Var(self.t, within=Reals, bounds=(0, None))

        self._model.power_balance = ConstraintList()
        for t in self.t:
            self._model.power_balance.add(0 == self._model.power_grid[t] + generation[t]
                                          - self._model.power_wp[t]
                                          - demand_power[t])
            self._model.power_balance.add(self._model.power_grid[t] ==
                                          self._model.power_grid_out[t] - self._model.power_grid_in[t])

        self._model.heat_to_power = ConstraintList()
        for t in self.t:
            self._model.heat_to_power.add(self._model.power_wp[t] == self._model.heat_wp_hw[t] / cop_hw[t]
                                          + self._model.heat_wp_sh[t] / cop_sh[t])

        self._model.volume_constraint_hw = ConstraintList()
        for t in self.t:
            if t > 0:
                self._model.volume_constraint_hw.add(self._model.volume_hw[t] ==
                                                     self._model.volume_hw[t-1]
                                                     + (self._model.v_in_hw[t] - self._model.v_out_hw[t]) * self.dt)
            else:
                self._model.volume_constraint_hw.add(self._model.volume_hw[t] ==
                                                     (self._model.v_in_hw[t] - self._model.v_out_hw[t]) * self.dt)

        self._model.volume_constraint_sh = ConstraintList()
        for t in self.t:
            if t > 0:
                self._model.volume_constraint_sh.add(self._model.volume_sh[t] ==
                                                     self._model.volume_sh[t-1]
                                                     + (self._model.v_in_sh[t] - self._model.v_out_sh[t]) * self.dt)
            else:
                self._model.volume_constraint_sh.add(self._model.volume_sh[t] ==
                                                     (self._model.v_in_sh[t] - self._model.v_out_sh[t]) * self.dt)

        self._model.demand_hw = ConstraintList()
        for t in self.t:
            self._model.demand_hw.add(demand_hw[t] + self._model.v_in_hw[t] ==
                                      self._model.heat_wp_hw[t] + self._model.v_out_hw[t])

        self._model.demand_sh = ConstraintList()
        for t in self.t:
            self._model.demand_sh.add(demand_sh[t] + self._model.v_in_sh[t] ==
                                      self._model.heat_wp_sh[t] + self._model.v_out_sh[t])

        self._model.objective = Objective(expr=quicksum(40 * self._model.power_grid_out[t] * self.dt
                                                        + 7* self._model.power_grid_in[t] * self.dt
                                                        for t in self.t), sense=minimize)

        self._solver.solve(self._model)

        return self._model


if __name__ == "__main__":
    from config import SimulationConfig
    from matplotlib import pyplot as plt

    Config = SimulationConfig()
    config_dict = Config.get_config_dict()
    date_range = pd.date_range(start=Config.START_DATE, end=Config.END_DATE + td(days=1), freq="d")
    time_range = pd.date_range(start=Config.START_DATE, end=Config.END_DATE + td(days=1), freq=Config.RESOLUTION)[:-1]

    WeatherGenerator = WeatherGenerator()
    # -> get weather data
    weather_at_each_day = []
    for day in date_range:
        weather_at_each_day.append(WeatherGenerator.get_weather(day))
    weather_at_each_day = pd.concat(weather_at_each_day)
    # -> resample to quarter hour values
    weather_at_each_day = weather_at_each_day.resample("15min").ffill()
    # -> remove not used time steps
    idx = weather_at_each_day.index.isin(list(time_range))
    weather_at_each_day = weather_at_each_day.loc[idx]
    config_dict['tariff'] = None
    my_household = HouseholdWithHeat(
        residents=5,
        demand_power=6_000,
        demand_heat=15_000,
        random=np.random.default_rng(0),
        pv_systems=[{'pdc0': 2.1, 'surface_tilt': 33, 'surface_azimuth': 158}],
        weather=weather_at_each_day,
        **config_dict
    )

    heat_values = my_household.get_heat_demand(date_range[0])
    demand_hw = heat_values['hot_water']
    cop_hw = heat_values['cop_hot_water']

    demand_sh = heat_values['space_heating']
    cop_sh = heat_values['space_heating']
    generation = my_household.get_pv_generation(date_range[0])

    m = my_household.optimize(date_range[0])
    steps = np.arange(96)

    wp_hw = [m.heat_wp_hw[t].value for t in steps]
    plt.plot(wp_hw)
    v_in_hw = [m.v_in_hw[t].value for t in steps]
    plt.plot(v_in_hw)
    v_out_hw = [m.v_out_hw[t].value for t in steps]
    volume_hw = [m.volume_hw[t].value for t in steps]

    plt.plot(v_out_hw)

    plt.plot(demand_hw)

    plt.plot(volume_hw)

    plt.plot(generation.values)

    plt.legend(['WP', 'V_in', 'V_out', 'demand', 'volume', 'pv'])



    # grid = [m.power_grid[t].value for t in steps]
    # plt.plot(grid)

