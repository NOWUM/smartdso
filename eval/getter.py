from sqlalchemy import create_engine, inspect
import pandas as pd
import numpy as np
import os

DATABASE_URI = os.getenv('DATABASE_URI', 'postgresql://opendata:opendata@10.13.10.41:5432/smartdso')


class EvalGetter:

    def __init__(self, database_uri: str = DATABASE_URI):
        self.engine = create_engine(database_uri)

        self._prices = pd.read_csv(r'./participants/data/2022_prices.csv', index_col=0, parse_dates=True)
        self._prices = self._prices.resample('15min').ffill()
        self._prices['price'] /= 10

        def get_scenarios():
            query = 'select distinct scenario from electric_vehicle ' \
                    'where time=(select time from electric_vehicle order by time limit 1) ' \
                    'and iteration = 0 order by scenario'
            data = pd.read_sql(query, self.engine)
            return list(data['scenario'])

        def get_cars(scenario: str):
            query = f"select distinct id_ from electric_vehicle where scenario='{scenario}' " \
                    f"and time=(select time from electric_vehicle order by time limit 1) order by id_"
            data = pd.read_sql(query, self.engine)
            return list(data['id_'])

        def get_time_range(scenario: str):
            query = f"select distinct time from grid_summary where scenario='{scenario}' and iteration = 0" \
                    f"order by time"
            data = pd.read_sql(query, self.engine)
            return list(data['time'])

        self.tables = inspect(self.engine).get_table_names()
        self.scenarios = get_scenarios()
        self.cars = {sc: get_cars(sc) for sc in self.scenarios}
        self.time_ranges = {sc: get_time_range(sc) for sc in self.scenarios}

    def get_all_utilization_values(self, scenario: str, asset: str = 'transformer'):
        table = 'grid_asset'
        query = f"select utilization from {table} " \
                f"where scenario = '{scenario}' and asset = '{asset}' order by utilization desc"
        data = pd.read_sql(query, self.engine)
        return data

    def get_mean_values(self, parameter: str, scenario: str, date_range: pd.DatetimeIndex = None):
        if parameter == 'market_prices':
            prices = self._prices.loc[date_range]
            return prices
        elif parameter == 'charging':
            insert = "sum(final_grid) + sum(final_pv) as charging"
        elif parameter in ['availability', 'grid_fee']:
            insert = f"avg({parameter}) as {parameter}"
        else:
            insert = f"sum({parameter}) as {parameter}"

        query = f"""select i_table.time, iteration, avg(i_table.{parameter}) as {parameter} from (
                        select time, iteration, {insert} from charging_summary where scenario = '{scenario}'
                        group by time, iteration order by time
                        ) i_table group by i_table.time, i_table.iteration
                """
        data = pd.read_sql(query, self.engine, index_col='time')

        return data.pivot(columns='iteration', values=parameter)

    def get_ev(self, scenario: str, ev: str):
        query = f"select time, avg(usage) as usage, avg(final_charging) as charging, " \
                f"(1-avg(usage)) * avg(pv) as used_pv_generation, avg(soc) as soc " \
                f"from electric_vehicle where id_='{ev}' and scenario='{scenario}' group by time order by time"

        data = pd.read_sql(query, self.engine, index_col='time')
        return data

    def get_aggregated_utilization_values(self, scenario: str, iteration: int = None, func='max',
                                          asset: str = 'transformer') -> pd.DataFrame:
        """average utilization through all sub grids"""
        if func == 'max':
            name = func
            func = name
        else:
            name = func
            func = 'avg'

        if iteration is None:
            query = f"""select time, sub_id, {func}(value) as util
                    from grid_summary where scenario = '{scenario}' and type='{name}' and asset='{asset}' 
                    group by sub_id, time order by time
                    """
        else:
            query = f"""select time, sub_id, {func}(value) as util from grid_summary
                    where scenario = '{scenario}' and type='{name}' and iteration={iteration} and asset='{asset}' 
                    group by sub_id, time order by time
                    """

        data = pd.read_sql(query, self.engine, index_col='time')

        return data.pivot(columns='sub_id', values='util')


# def get_auslastung(scenario: str, asset:str = 'line'):
#     query = f"select time, avg(utilization), id_ from grid_asset where asset='{asset}' and scenario='{scenario}' group by time order by time desc limit 1"
#     data = pd.read_sql(query, ENGINE)
#     return data
#
#
# def get_typ_values(parameter: str, scenario: str, date_range: pd.DatetimeIndex = None):
#
#     if parameter == 'market_prices':
#         prices = PRICES.loc[date_range]
#         data = prices.groupby(prices.index.time)['price'].mean()
#         data.index = data.index.astype(str)
#         return data
#     elif parameter == 'charging':
#         insert = "avg(final_grid) + avg(final_pv) as charging "
#     else:
#         insert = f"avg({parameter}) as {parameter} "
#
#     query = f"select to_char(time, 'hh24:mi') as interval, {insert}" \
#             f"from charging_summary where scenario = '{scenario}' group by interval"
#
#     data = pd.read_sql(query, ENGINE, index_col='interval')
#     return data

# def get_total_values(parameter: str, scenario: str):
#     table = 'charging_summary'
#     factor = 1
#     insert = f"sum({parameter}) as {parameter}"
#     if parameter in ['charging', 'final_grid', 'final_pv', 'initial_grid']:
#         if parameter == 'charging':
#             insert = "sum(final_grid) + sum(final_pv) as charging"
#         factor = 0.25 / 1e3
#     elif parameter == 'availability':
#         insert = f"avg({parameter}) as {parameter}"
#     elif parameter == 'distance':
#         table = 'electric_vehicle'
#
#     query = f"select iteration, {insert} from {table} where scenario='{scenario}' " \
#             f"group by iteration"
#
#     data = pd.read_sql(query, ENGINE, index_col='iteration')
#     data = data.mean() * factor
#
#     return data.values[0]
#
# def get_shifted(scenario: str):
#
#     query = f"""select iteration, sum(initial_grid - final_grid)
#         from charging_summary
#         where initial_grid > final_grid and scenario='{scenario}'
#         group by iteration
#         """
#     data = pd.read_sql(query, ENGINE, index_col='iteration')
#
#     return data
#
#


# def pv_capacity():
#     total_alloc = pd.read_csv(fr'./gridLib/data/grid_allocations.csv', index_col=0)
#     tc = total_alloc.dropna()
#     tc = tc['pv']
#     summ_pdc_alloc = 0
#     for val in tc:
#         l = eval(val)
#         for i in l:
#             summ_pdc_alloc += i['pdc0']
#
#     total_consumers = pd.read_csv(fr'./gridLib/data/export/dem/consumers.csv', index_col=0)
#     tc = total_consumers.dropna()
#     tc = tc['pv']
#     summ_pdc_consumer = 0
#     for val in tc:
#         l = eval(val)
#         for i in l:
#             summ_pdc_consumer += i['pdc0']
#     return summ_pdc_alloc, summ_pdc_consumer # [kWp]
#
# def get_max_power(scenario: str) -> float:
#     query = f"select id_, max(final_charging) as max_power from electric_vehicle where scenario = '{scenario}' group by id_"
#     data = pd.read_sql(query, ENGINE, index_col='id_')
#     return float(data.sum()) # [kW]
#
# def get_gzf(scenario: str, typ='power', typ_tage = False):
#     if typ_tage:
#         typ_agg = "to_char(time, 'hh24:mi')"
#         q = f"select count(*) from (select distinct time from charging_summary where scenario = '{scenario}') a"
#         days = int(pd.read_sql(q, ENGINE)['count'])/96
#     else:
#         typ_agg = 'time'
#         days = 1
#
#     if typ =='power':
#         query = f"""
#             select {typ_agg} as interval, sum(final_grid + final_pv) as charging
#             from charging_summary where scenario = '{scenario}'
#             group by interval order by interval
#             """
#     elif typ=='count':
#         query = f"""select {typ_agg} as interval, count(CASE WHEN final_charging > 0 THEN 1 END) as gzf
#             from electric_vehicle
#             where scenario = '{scenario}' group by interval order by interval
#             """
#     data = pd.read_sql(query, ENGINE, index_col='interval')
#
#     if typ =='power':
#         data['gzf'] = data['charging']/get_max_power(scenario)
#     else:
#         count = len(get_cars(scenario))
#         data = data/count
#     return data/days
