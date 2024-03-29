from tqdm import tqdm
import pandas as pd
from sqlalchemy import create_engine, inspect


# CREATE EXTENSION postgis;

DATABASE_HPFC = 'postgresql://opendata:opendata@10.13.10.41:5432/priceit'


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

    def __init__(self, database_uri: str, create_tables: bool = False):
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

        # -> electric vehicle table
        self.engine.execute("CREATE TABLE IF NOT EXISTS electric_vehicle( "
                            "time timestamp without time zone NOT NULL, "
                            "iteration integer, "
                            "scenario text, "
                            "sub_id integer, "
                            "id_ text, "
                            "soc double precision, "
                            "usage integer, "
                            "initial_charging double precision, "
                            "final_charging double precision, "
                            "demand double precision, "
                            "distance double precision, "
                            "pv double precision, "
                            "pv_available double precision, "
                            "tariff double precision, "
                            "PRIMARY KEY (time , scenario, iteration, sub_id, id_));")

        query_create_hypertable = "SELECT create_hypertable('electric_vehicle', 'time', if_not_exists => TRUE, migrate_data => TRUE);"
        with self.engine.connect() as connection:
            with connection.begin():
                connection.execute(query_create_hypertable)

        # -> create heat demand table
        self.engine.execute("CREATE TABLE IF NOT EXISTS heat_demand( "
                            "time timestamp without time zone NOT NULL, "
                            "iteration integer, "
                            "scenario text, "
                            "id_ text, "
                            "sub_id integer, "
                            "demand_hot_water double precision, "
                            "cop_hot_water double precision, "
                            "demand_space_heating double precision, "
                            "cop_space_heating double precision, "
                            "volume_space_heating double precision, "
                            "volume_hot_water double precision, "
                            "power_heat_pump double precision, "
                            "PRIMARY KEY (time , scenario, iteration, sub_id, id_));")

        query_create_hypertable = "SELECT create_hypertable('heat_demand', 'time', if_not_exists => TRUE, migrate_data => TRUE);"
        with self.engine.connect() as connection:
            with connection.begin():
                connection.execute(query_create_hypertable)

        # -> grid asset table
        self.engine.execute("CREATE TABLE IF NOT EXISTS grid_asset( "
                            "time timestamp without time zone NOT NULL, "
                            "iteration integer, "
                            "scenario text, "
                            "utilization double precision, "
                            "id_ text, "
                            "sub_id integer, "
                            "asset text, "
                            "PRIMARY KEY (time , scenario, iteration, id_));")

        query_create_hypertable = "SELECT create_hypertable('grid_asset', 'time', if_not_exists => TRUE, migrate_data => TRUE);"
        with self.engine.connect() as connection:
            with connection.begin():
                connection.execute(query_create_hypertable)

        # -> charging table
        self.engine.execute("CREATE TABLE IF NOT EXISTS charging_summary( "
                            "time timestamp without time zone NOT NULL, "
                            "iteration integer, "
                            "scenario text, "
                            "sub_id integer, "
                            "initial_grid double precision, "
                            "final_grid double precision, "
                            "final_pv double precision, "
                            "car_demand double precision, "
                            "residual_generation double precision, "
                            "residential_demand double precision, "
                            "availability double precision, "
                            "grid_fee double precision, "
                            "PRIMARY KEY (time , scenario, iteration, sub_id));")

        query_create_hypertable = "SELECT create_hypertable('charging_summary', 'time', if_not_exists => TRUE, migrate_data => TRUE);"
        with self.engine.connect() as connection:
            with connection.begin():
                connection.execute(query_create_hypertable)

        # -> grid table
        self.engine.execute("CREATE TABLE IF NOT EXISTS grid_summary( "
                            "time timestamp without time zone NOT NULL, "
                            "iteration integer, "
                            "scenario text, "
                            "asset text,"
                            "type text , "
                            "sub_id integer,"
                            "value double precision, "
                            "PRIMARY KEY (time , scenario, iteration, type, asset, sub_id));")

        query_create_hypertable = "SELECT create_hypertable('grid_summary', 'time', if_not_exists => TRUE, migrate_data => TRUE);"
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
        for table in tqdm(self.tables):
            query = f"DELETE FROM {table} WHERE scenario='{scenario}';"
            try:
                self.engine.execute(query)
            except Exception as e:
                pass
                # print(repr(e))

    def delete_time_range(self, t: str, sign: str = '>'):
        for table in self.tables:
            query = f"DELETE FROM {table} WHERE time {sign} '{t}';"
            try:
                self.engine.execute(query)
            except Exception as e:
                print(repr(e))


if __name__ == "__main__":
    tb = TableCreator(create_tables=False, database_uri='postgresql://opendata:opendata@10.13.10.41:5432/smartdso')
    for scenario in ['Alliander_HPEV_Test', 'Alliander_HP1_Test']:
        tb.delete_scenario(scenario)
    # prefix = 'A-'
    # scenarios = [
    #     f'{prefix}MaxPvCap-PV25-PriceFlat',
    #     f'{prefix}MaxPvSoc-PV80-PriceSpot',
    #     f'{prefix}MaxPvCap-PV80-PriceSpot',
    #     f'{prefix}PlugInCap-PV25-PriceFlat'
    # ]
    #
    # for scenario in scenarios:
    #     tb.delete_scenario(scenario)