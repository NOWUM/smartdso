from sqlalchemy import create_engine, inspect
from functools import lru_cache
import pandas as pd
import os


DATABASE_URI = os.getenv('DATABASE_URI', 'postgresql://opendata:opendata@10.13.10.41:5432/smartdso')

class Results:

    def __init__(self, create_tables: bool = False):
        self.engine = create_engine(DATABASE_URI)
        if create_tables:
            self._create_tables()
        self.tables = inspect(self.engine).get_table_names()
        self.scenarios = self._get_scenarios()

    def _create_tables(self):
        # -> grid table
        self.engine.execute("CREATE TABLE IF NOT EXISTS grid( "
                            "time timestamp without time zone NOT NULL, "
                            "iteration integer, "
                            "scenario text, "
                            "id_ text, "
                            "sub_id integer , "
                            "asset text, "
                            "utilization double precision, "
                            "PRIMARY KEY (time , id_, iteration, scenario));")

        query_create_hypertable = "SELECT create_hypertable('grid', 'time', if_not_exists => TRUE, migrate_data => TRUE);"
        with self.engine.connect() as connection:
            with connection.begin():
                connection.execute(query_create_hypertable)
        # -> variables table
        self.engine.execute("CREATE TABLE IF NOT EXISTS vars( "
                            "time timestamp without time zone NOT NULL, "
                            "iteration integer, "
                            "scenario text, "
                            "charged double precision, "
                            "shifted double precision, "
                            "price double precision, "
                            "soc double precision, "
                            "cost double precision, "
                            "total_ev double precision, "
                            "avg_distance double precision, "
                            "avg_demand double precision, "
                            "sub_id integer, "
                            "PRIMARY KEY (time , iteration, scenario));")

        query_create_hypertable = "SELECT create_hypertable('vars', 'time', if_not_exists => TRUE, migrate_data => TRUE);"
        with self.engine.connect() as connection:
            with connection.begin():
                connection.execute(query_create_hypertable)
        # -> car table
        self.engine.execute("CREATE TABLE IF NOT EXISTS cars( "
                            "time timestamp without time zone NOT NULL, "
                            "iteration integer, "
                            "scenario text, "
                            "distance double precision, "
                            "odometer double precision, "
                            "soc double precision, "
                            "work integer, "
                            "errand integer, "
                            "hobby integer, "
                            "PRIMARY KEY (time , iteration, scenario));")

        query_create_hypertable = "SELECT create_hypertable('cars', 'time', if_not_exists => TRUE, migrate_data => TRUE);"
        with self.engine.connect() as connection:
            with connection.begin():
                connection.execute(query_create_hypertable)

    def _get_scenarios(self):
        scenarios = dict()
        for table in self.tables[:1]:
            query = f'select distinct scenario from {table}'
            scenarios[table] = set([value._data[0] for value in self.engine.execute(query).fetchall()])
        return set.intersection(*scenarios.values()) if len(scenarios) > 0 else set()

    def delete_scenario(self, scenario: str):
        for table in self.tables:
            query = f"delete from {table} where scenario='{scenario}'"
            self.engine.execute(query)

    def _get_iterations(self, scenario: str):
        iteration = dict()
        for table in self.tables:
            query = f"select distinct iteration from {table} where scenario='{scenario}'"
            iteration[table] = set([value._data[0] for value in self.engine.execute(query).fetchall()])
        return set.intersection(*iteration.values()) if len(iteration) > 0 else set()

    @lru_cache
    def get_cars(self, scenario: str, iteration: int):
        query = f"Select time, odometer, soc, work, errand, hobby from cars where scenario='{scenario}' " \
                f"and iteration={iteration}"
        dataframe = pd.read_sql(query, self.engine).set_index('time')
        dataframe.index = pd.to_datetime(dataframe.index)
        dataframe = dataframe.sort_index()

        return dataframe

    @lru_cache
    def get_evs(self, scenario: str, iteration: int):
        query = f"select total_ev, avg_distance, avg_demand from vars where scenario='{scenario}' " \
                f"and iteration={iteration}"
        dataframe = pd.read_sql(query, self.engine)

        return dataframe.iloc[0, :]

    @lru_cache
    def get_vars(self, scenario: str):
        query = f"select time, avg(price) as avg_price, max(price) as max_price, min(price) as min_price, " \
                f"avg(shifted) as avg_shifted, max(shifted) as max_shifted, min(shifted) as min_shifted," \
                f"avg(charged) as charged from vars where scenario='{scenario}' " \
                f"group by time order by time"
        dataframe = pd.read_sql(query, self.engine).set_index('time')
        dataframe.index = pd.to_datetime(dataframe.index)
        for col in dataframe.columns:
            dataframe[col] = dataframe[col].apply(lambda x: round(x, 2))

        return dataframe

    @lru_cache
    def get_asset_type_util(self, asset: str, scenario: str):
        query = f"select to_char(time, 'hh24:00') as t, " \
                f"percentile_cont(0.25) within group (order by utilization) as percentile_25," \
                f" percentile_cont(0.75) within group (order by utilization) as percentile_75, " \
                f" percentile_cont(0.95) within group (order by utilization) as percentile_95, " \
                f"avg(utilization), max(utilization) "
        if asset == 'transformer' or asset == 'outlet' or asset == 'inlet':
            query += f"from grid where scenario='{scenario}' and asset='{asset}' group by t"
        elif asset == 'line':
            query += f"from grid where scenario='{scenario}' and (asset='outlet' " \
                     f"or asset='inlet') group by t"

        dataframe = pd.read_sql(query, self.engine)
        dataframe = dataframe.set_index(['t'])
        dataframe.columns = ['Percentile 25 %', 'Percentile 75 %', 'Percentile 95 %', 'Average', 'Maximum']

        return dataframe

    @lru_cache
    def get_sorted_utilization(self, scenario: str):
        query = f"select utilization as util from grid where scenario='{scenario}' " \
                f"and utilization > 0.001 order by utilization desc"
        dataframe = pd.read_sql(query, self.engine)
        return dataframe.values.flatten()

    @lru_cache
    def get_line_utilization(self, scenario: str, iteration: int, t: str):
        query = f"select id_, utilization from grid where scenario='{scenario}' " \
                f"and iteration='{iteration}' and (asset='outlet' or asset='inlet') " \
                f"and time ='{t}'"
        dataframe = pd.read_sql(query, self.engine)
        dataframe.columns = ['name', 'utilization']
        return dataframe.set_index('name')

