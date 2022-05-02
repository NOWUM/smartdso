from sqlalchemy import create_engine, inspect
import pandas as pd


class Results:

    def __init__(self):
        self.engine = create_engine('postgresql://opendata:opendata@10.13.10.41:5432/smartdso')
        self._create_tables()
        self.tables = inspect(self.engine).get_table_names()
        self.scenarios = self._get_scenarios()
        if len(self.scenarios) > 0:
            self.scenario = [*self.scenarios][0]
            self.iterations = self._get_iterations()
            self.iteration = [*self.iterations][0]
        else:
            self.scenario = None
            self.iterations = None

    def _create_tables(self):
        # -> grid table
        self.engine.execute("CREATE TABLE IF NOT EXISTS grid( "
                            "time timestamp without time zone NOT NULL, "
                            "iteration integer, "
                            "scenario text, "
                            "id_ text, "
                            "sub_id integer , "
                            "asset text, "
                            "avg_util double precision, "
                            "max_util double precision, "
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
        for table in self.tables:
            query = f'Select Distinct scenario from {table}'
            scenarios[table] = set([value._data[0] for value in self.engine.execute(query).fetchall()])
        return set.intersection(*scenarios.values()) if len(scenarios) > 0 else set()

    def delete_scenario(self, scenario: str):
        for table in self.tables:
            query = f"DELETE FROM {table} WHERE scenario='{scenario}'"
            self.engine.execute(query)

    def _get_iterations(self):
        iteration = dict()
        for table in self.tables:
            query = f"Select Distinct iteration from {table} where scenario='{self.scenario}'"
            iteration[table] = set([value._data[0] for value in self.engine.execute(query).fetchall()])
        return set.intersection(*iteration.values()) if len(iteration) > 0 else set()

    def get_cars(self):
        query = f"Select time, odometer, soc, work, errand, hobby from cars where scenario='{self.scenario}' " \
                f"and iteration={self.iteration}"
        dataframe = pd.read_sql(query, self.engine).set_index('time')
        dataframe.index = pd.to_datetime(dataframe.index)
        dataframe = dataframe.sort_index()

        return dataframe

    def get_evs(self):
        query = f"select total_ev, avg_distance, avg_demand from vars where scenario='{self.scenario}' " \
                f"and iteration={self.iteration}"
        dataframe = pd.read_sql(query, self.engine)

        return dataframe.iloc[0, :]

    def get_vars(self):
        query = f"select time, avg(price) as avg_price, max(price) as max_price, min(price) as min_price, " \
                f"avg(shifted) as avg_shifted, max(price) as max_shifted, min(shifted) as min_shifted," \
                f"avg(charged) as charged from vars where scenario='{self.scenario}' " \
                f"group by time order by time"
        dataframe = pd.read_sql(query, self.engine).set_index('time')
        dataframe.index = pd.to_datetime(dataframe.index)
        dataframe = dataframe.resample('5min').mean()
        for col in dataframe.columns:
            dataframe[col] = dataframe[col].apply(lambda x: round(x, 2))

        return dataframe

    def get_maximal_util(self, asset: str):
        query = f"select total.id_, 100*filter.c::decimal/total.c::decimal as quantil, total.util " \
                f"from (select id_, count(id_) as c, avg(avg_util) as util from grid " \
                f"where scenario='{self.scenario}' and asset='{asset}' group by grid.id_) as total " \
                f"join (select id_, count(id_) as c from grid " \
                f"where max_util > 75 and scenario='{self.scenario}' and asset='{asset}' " \
                f"group by grid.id_) as filter " \
                f"on filter.id_ = total.id_ order by quantil desc limit 5"

        dataframe = pd.read_sql(query, self.engine).set_index(['id_'])
        dataframe = dataframe.sort_values(['util'], ascending=False)
        dataframe.columns = [f'>75 %', 'Mean Utilization %']
        for col in dataframe.columns:
            dataframe[col] = dataframe[col].apply(lambda x: round(x, 2))

        return dataframe

    def get_asset_type_util(self, asset: str):

        query = f"select to_char(time, 'YYYY-MM-DD hh24:00:00') as t, avg_util, id_ " \
                f"from grid where scenario='{self.scenario}' and asset='{asset}'"
        dataframe = pd.read_sql(query, self.engine)
        dataframe['t'] = dataframe['t'].map(pd.to_datetime)
        utilization = [dataframe.loc[dataframe['t'].dt.hour == h, 'avg_util'].values.flatten() for h in range(24)]

        return utilization

    def get_asset_util(self, id_: str):
        query = f"select id_, max(max_util) from grid where id_='{id_}' and scenario='{self.scenario}' group by id_"
        dataframe = pd.read_sql(query, self.engine)
        return dataframe

#
# def get_transformers(scenario: str, sub_id: str):
#     # -> get the five with the highest utilization
#     if sub_id != 'total':
#         query = f"select transformers.id_, time, utilization as util from transformers " \
#                 f"join (select distinct id_, max(utilization) as util from transformers where scenario='{scenario}' " \
#                 f"and grid='{sub_id}'" \
#                 f"group by id_ order by util desc limit 5) " \
#                 f"as first5 on first5.id_=transformers.id_ and first5.util=transformers.utilization"
#     else:
#         query = f"select transformers.id_, time, utilization as util from transformers " \
#                 f"join (select distinct id_, max(utilization) as util from transformers where scenario='{scenario}' " \
#                 f"group by id_ order by util desc limit 5) " \
#                 f"as first5 on first5.id_=transformers.id_ and first5.util=transformers.utilization"
#
#     max_utilization = pd.read_sql(query, engine).set_index('time')
#     return max_utilization
#
#
# def get_lines(scenario: str, sub_id: str):
#     # -> get the five with the highest utilization
#     if sub_id != 'total':
#         query = f"select lines.id_, time, utilization as util from lines " \
#                 f"join (select distinct id_, max(utilization) as util from lines where scenario='{scenario}' " \
#                 f"and grid='{sub_id}'" \
#                 f"group by id_ order by util desc limit 5) " \
#                 f"as first5 on first5.id_=lines.id_ and first5.util=lines.utilization"
#     else:
#         query = f"select lines.id_, time, utilization as util from lines " \
#                 f"join (select distinct id_, max(utilization) as util from lines where scenario='{scenario}' " \
#                 f"group by id_ order by util desc limit 5) " \
#                 f"as first5 on first5.id_=lines.id_ and first5.util=lines.utilization"
#
#     max_utilization = pd.read_sql(query, engine).set_index('time')
#
#     if sub_id != 'total':
#         query = f"Select id_, max(utilization) as utilization from lines where scenario='{scenario}' and grid='{sub_id}' " \
#                 f"group by id_"
#     else:
#         query = f"Select id_, max(utilization) as utilization from lines where scenario='{scenario}' group by id_"
#
#     total_utilization = pd.read_sql(query, engine).set_index('id_')
#
#     return max_utilization, total_utilization
#
#
# def get_utilization_distribution(scenario: str, limit: int):
#     query = f"select total.id_, 100*filter.c::decimal/total.c::decimal as quantil, total.util " \
#             f"from (select id_, count(id_) as c, avg(utilization) as util from lines " \
#             f"where scenario='{scenario}' group by lines.id_) as total " \
#             f"join (select id_, count(id_) as c from lines " \
#             f"where utilization > {limit} and scenario='{scenario}' " \
#             f"group by lines.id_) as filter " \
#             f"on filter.id_ = total.id_ order by quantil desc limit 5"
#
#     table = pd.read_sql(query, engine).set_index(['id_'])
#     table = table.sort_values(['util'], ascending=False)
#     table.columns = [f'>{limit} %', 'Mean Utilization %']
#
#     return table
#
#
# def get_transformer_utilization(scenario: str, sub_id: str):
#     query = f"Select time, avg(utilization) as avg, min(utilization) as min, max(utilization) as max " \
#             f"from transformers where scenario='{scenario}' and grid='{sub_id}' group by time order by time"
#
#     data = pd.read_sql(query, engine).set_index('time')
#
#     for column in data.columns:
#         data[column] = data[column].apply(lambda x: round(x, 2))
#
#     return data
#
#
#
# def get_x(scenario: str, limit=30):
#     query = f"select to_char(time, 'YYYY-MM-DD hh24:00:00') as t, count(utilization) " \
#             f"from lines where scenario='{scenario}' and utilization >= {limit} group by t order by t "
#
#     dataframe = pd.read_sql(query, engine).set_index(['t'])
#     dataframe.index = dataframe.index.map(pd.to_datetime)
#     dataframe.columns = [limit]
#     return dataframe


if __name__ == "__main__":
    r = Results()
    df_sim = r.get_vars()
    df_car = r.get_cars()
    df_util = r.get_maximal_util(asset='outlet')
    df_evs = r.get_evs()
    util = r.get_asset_type_util(asset='outlet')

