from sqlalchemy import create_engine, inspect
import os

DATABASE_URI = os.getenv('DATABASE_URI', 'postgresql://opendata:opendata@10.13.10.41:5432/smartgrid')


class TableCreator:

    def __init__(self, create_tables: bool = False, database_uri: str = DATABASE_URI):
        self.engine = create_engine(database_uri)
        self.tables = inspect(self.engine).get_table_names()
        if create_tables:
            for table in self.tables:
                if table != 'spatial_ref_sys':
                    print(table)
                    self.engine.execute(f"DROP TABLE IF EXISTS {table}")
            self._create_tables()
            self.tables = inspect(self.engine).get_table_names()

    def _create_tables(self):
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

        # -> residential table
        self.engine.execute("CREATE TABLE IF NOT EXISTS residential( "
                            "time timestamp without time zone NOT NULL, "
                            "consumer_id text, "
                            "node_id text,"
                            "iteration integer, "
                            "scenario text, "
                            "demand double precision, "
                            "residual_demand double precision, "
                            "generation double precision, "
                            "residual_generation double precision, "
                            "pv_capacity double precision, "
                            "total_radiation double precision, "
                            "tariff double precision, "
                            "grid_fee double precision, "
                            "car_demand double precision, "
                            "car_capacity double precision, "
                            "planed_grid_consumption double precision, "
                            "final_grid_consumption double precision, "
                            "planed_pv_consumption double precision, "
                            "final_pv_consumption double precision, "
                            "PRIMARY KEY (time , consumer_id, iteration, scenario));")

        query_create_hypertable = "SELECT create_hypertable('residential', 'time', if_not_exists => TRUE, migrate_data => TRUE);"
        with self.engine.connect() as connection:
            with connection.begin():
                connection.execute(query_create_hypertable)

        # -> car table
        self.engine.execute("CREATE TABLE IF NOT EXISTS cars( "
                            "time timestamp without time zone NOT NULL, "
                            "car_id text, "
                            "consumer_id text, "
                            "iteration integer, "
                            "scenario text, "
                            "distance double precision, "
                            "total_distance double precision, "
                            "planed_charge double precision, "
                            "final_charge double precision, "
                            "demand double precision, "
                            "soc double precision, "
                            "usage integer,"
                            "work integer, "
                            "errand integer, "
                            "hobby integer, "
                            "PRIMARY KEY (time , car_id, iteration, scenario));")

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
