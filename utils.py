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

        self.engine.execute("CREATE EXTENSION postgis;")

        # -> grid table
        self.engine.execute("CREATE TABLE IF NOT EXISTS grid( "
                            "time timestamp without time zone NOT NULL, "
                            "iteration integer, "
                            "scenario text, "
                            "id_ text, "
                            "sub_grid integer , "
                            "asset text, "
                            "utilization double precision, "
                            "PRIMARY KEY (time , id_, iteration, scenario));")

        query_create_hypertable = "SELECT create_hypertable('grid', 'time', if_not_exists => TRUE, migrate_data => TRUE);"
        with self.engine.connect() as connection:
            with connection.begin():
                connection.execute(query_create_hypertable)

        self.engine.execute("CREATE INDEX grid_id ON grid (id_);")
        self.engine.execute("CREATE INDEX grid_iteration ON grid (iteration);")
        self.engine.execute("CREATE INDEX grid_scenario ON grid (scenario);")

        # -> residential table
        self.engine.execute("CREATE TABLE IF NOT EXISTS residential( "
                            "time timestamp without time zone NOT NULL, "
                            "consumer_id text, "
                            "node_id text,"
                            "iteration integer, "
                            "scenario text, "
                            "demand double precision, "
                            "generation double precision, "
                            "pv_capacity double precision, "
                            "grid_fee double precision, "
                            "car_demand double precision, "
                            "car_capacity double precision, "
                            "planned_pv_consumption double precision, "
                            "final_pv_consumption double precision, "
                            "PRIMARY KEY (time , consumer_id, iteration, scenario));")

        query_create_hypertable = "SELECT create_hypertable('residential', 'time', if_not_exists => TRUE, migrate_data => TRUE);"
        with self.engine.connect() as connection:
            with connection.begin():
                connection.execute(query_create_hypertable)

        self.engine.execute("CREATE INDEX residential_id ON residential (consumer_id);")
        self.engine.execute("CREATE INDEX residential_iteration ON residential (iteration);")
        self.engine.execute("CREATE INDEX residential_scenario ON residential (scenario);")

        # -> car table
        self.engine.execute("CREATE TABLE IF NOT EXISTS cars( "
                            "time timestamp without time zone NOT NULL, "
                            "car_id text, "
                            "consumer_id text, "
                            "iteration integer, "
                            "scenario text, "
                            "distance double precision, "
                            "total_distance double precision, "
                            "planned_charge double precision, "
                            "final_charge double precision, "
                            "demand double precision, "
                            "soc double precision, "
                            "usage integer,"
                            "PRIMARY KEY (time , car_id, iteration, scenario));")

        self.engine.execute("CREATE INDEX car_id ON cars (car_id);")
        self.engine.execute("CREATE INDEX car_iteration ON cars (iteration);")
        self.engine.execute("CREATE INDEX car_scenario ON cars (scenario);")

        query_create_hypertable = "SELECT create_hypertable('cars', 'time', if_not_exists => TRUE, migrate_data => TRUE);"
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
    tb = TableCreator(create_tables=False)
    tb.delete_scenario(scenario='EV100PV100PRC40.0STR-SPV')
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
