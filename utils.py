import numpy as np
import pandas as pd
from sqlalchemy import create_engine, inspect
import os

# CREATE EXTENSION postgis;

DATABASE_URI = os.getenv('DATABASE_URI', 'postgresql://opendata:opendata@10.13.10.41:5432/smartdso')
DATABASE_HPFC = os.getenv('DATABASE_URI', 'postgresql://opendata:opendata@10.13.10.41:5432/priceit')


class PriceIT:

    def __init__(self, table_name: str = 'price_simulation', drop: bool = False):
        self.engine = create_engine(DATABASE_HPFC)
        self.price_data = pd.DataFrame()
        self.table_name = table_name
        self.create_table(drop=drop)

    def create_table(self, drop: bool = False):
        if drop:
            self.engine.execute(f"DROP TABLE IF EXISTS {self.table_name}")
        self.engine.execute(f"CREATE TABLE IF NOT EXISTS {self.table_name}( "
                            f"time timestamp without time zone NOT NULL, "
                            f"simulation integer, "
                            f"price double precision, "
                            f"PRIMARY KEY (time , simulation));")

        query_create_hypertable = "SELECT create_hypertable('price_simulation', 'time', if_not_exists => TRUE, migrate_data => TRUE);"
        with self.engine.connect() as connection:
            with connection.begin():
                connection.execute(query_create_hypertable)

    def read_csv_data(self):
        self.price_data = pd.read_csv(r'agents/data/spotprice.csv', index_col=0, parse_dates=True)

    def export_price_simulation(self):
        if self.price_data.empty:
            self.read_csv_data()

        for col in self.price_data.columns:
            df = self.price_data.loc[:, [col]]
            df['simulation'] = int(col.replace('Var', ''))
            df.columns = ['price', 'simulation']
            df.index.name = 'time'
            df = df[~df.index.duplicated(keep='first')]
            df.to_sql(self.table_name, self.engine, if_exists='append')
            print(f'imported simulation {col}')

    def get_simulation(self, simulation: int = 1):
        query = f"select price from {self.table_name} where simulation={simulation} " \
                f"and time >= '2023-01-01'  and time <= '2024-01-01'"
        df = pd.read_sql(query, self.engine)
        return df['price'].values.flatten()


class TableCreator:

    def __init__(self, create_tables: bool = False, database_uri: str = DATABASE_URI):
        self.engine = create_engine(database_uri)
        self.tables = inspect(self.engine).get_table_names()
        if create_tables:
            for table in self.tables:
                if table != 'spatial_ref_sys':
                    self.engine.execute(f"DROP TABLE IF EXISTS {table}")
            self._create_tables()
            self.tables = inspect(self.engine).get_table_names()

    def _create_tables(self):

        self.engine.execute("CREATE EXTENSION IF NOT EXISTS postgis;")

        # -> charging table
        self.engine.execute("CREATE TABLE IF NOT EXISTS charging( "
                            "time timestamp without time zone NOT NULL, "
                            "iteration integer, "
                            "scenario text, "
                            "type text , "
                            "value double precision, "
                            "PRIMARY KEY (time , scenario, iteration, type));")

        query_create_hypertable = "SELECT create_hypertable('charging', 'time', if_not_exists => TRUE, migrate_data => TRUE);"
        with self.engine.connect() as connection:
            with connection.begin():
                connection.execute(query_create_hypertable)

        # -> grid table
        self.engine.execute("CREATE TABLE IF NOT EXISTS grid( "
                            "time timestamp without time zone NOT NULL, "
                            "iteration integer, "
                            "scenario text, "
                            "asset text,"
                            "type text , "
                            "sub_id integer,"
                            "value double precision, "
                            "PRIMARY KEY (time , scenario, iteration, type, asset, sub_id));")

        query_create_hypertable = "SELECT create_hypertable('grid', 'time', if_not_exists => TRUE, migrate_data => TRUE);"
        with self.engine.connect() as connection:
            with connection.begin():
                connection.execute(query_create_hypertable)

        # -> geojson table
        self.engine.execute("CREATE TABLE IF NOT EXISTS edges_geo( "
                            "name text, "
                            "geometry geometry(LineString, 4326), "
                            "asset text, "
                            "PRIMARY KEY (name));")

        # -> geojson table
        self.engine.execute("CREATE TABLE IF NOT EXISTS nodes_geo( "
                            "name text, "
                            "geometry geometry(Point, 4326), "
                            "asset text, "
                            "PRIMARY KEY (name));")

        # -> geojson table
        self.engine.execute("CREATE TABLE IF NOT EXISTS transformers_geo( "
                            "name text, "
                            "geometry geometry(Point, 4326), "
                            "asset text, "
                            "PRIMARY KEY (name));")

    def delete_scenario(self, scenario: str):
        for table in self.tables:
            query = f"DELETE FROM {table} WHERE scenario='{scenario}';"
            try:
                self.engine.execute(query)
            except Exception as e:
                print(repr(e))


if __name__ == "__main__":
    tb = TableCreator(create_tables=True)
    # tb.delete_scenario(scenario='EV100PV80PRCFlatSTRMaxPvCap')
    #tb.delete_scenario(scenario='EV100PV100PRC40.0STR-S')
    #tb.delete_scenario(scenario='EV100PV80PRC40.0STR-S')
    #tb.delete_scenario(scenario='EV100PV50PRC40.0STR-S')
    #tb.delete_scenario(scenario='EV100PV80PRC4.0STR-O')


    #price_it = PriceIT()
    # price_it.export_price_simulation()
    #num = 500
    #sims = [price_it.get_simulation(simulation=sim+1) for sim in range(num)]
    #sims = np.asarray(sims, dtype=float)
    # pTb.export_price_simulation()
