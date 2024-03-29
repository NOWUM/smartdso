import os
import pandas as pd
from dataclasses import dataclass

TIME_MAPPER: dict = {1440: "min", 96: "15min", 24: "h"}
steps: int = int(os.getenv("STEPS_PER_DAY", 96))


@dataclass
class SimulationConfig:
    # -> steps and corresponding time resolution strings in pandas
    RESOLUTION: str = TIME_MAPPER[steps]
    STEPS: int = steps
    # -> timescaledb connection to store the simulation results
    DATABASE_URI: str = os.getenv(
        "DATABASE_URI", "postgresql://opendata:opendata@10.13.10.41:5432/smartdso"
    )
    # -> select grid data in ./gridLib/data/export/
    GRID_DATA: str = os.getenv("GRID_DATA", "alliander")
    # -> select sub grid: -1 correspond to total grid, an integer > 0 select the grid
    SUB_GRID: int = int(os.getenv("SUB_GRID", 1))
    # -> default start date
    START_DATE: pd.Timestamp = pd.to_datetime(os.getenv("START_DATE", "2022-01-01"))
    # -> default end date
    END_DATE: pd.Timestamp = pd.to_datetime(os.getenv("END_DATE", "2022-01-10"))
    # -> set EV ratio
    EV_RATIO: int = int(os.getenv("EV_RATIO", 0)) / 100
    # -> set PV ratio
    PV_RATIO: int = int(os.getenv("PV_RATIO", 25)) / 100
    # -> set hp ratio
    HP_RATIO: int = int(os.getenv("HP_RATIO", 25)) / 100
    # -> use historic london data or slp data
    LONDON_DATA: bool = os.getenv("LONDON_DATA", "False") == "True"
    # -> set consumer charging strategy
    STRATEGY: str = os.getenv("STRATEGY", "optimize_soc")
    # -> set tariff for flexibility provider
    TARIFF: str = os.getenv("TARIFF", "spot")
    # -> set seed to avoid monte carlo error
    SEED: int = int(os.getenv("RANDOM_SEED", 2022))
    # -> set name for simulation
    NAME: str = os.getenv("NAME", "AlliandergridTest1")
    # -> set simulation number
    SIM: int = 0
    # -> reset/initialize database
    RESET_DATABASE: bool = os.getenv("RESET_DATABASE", "False") == "True"
    # -> delete scenario
    DELETE_SCENARIO: bool = os.getenv("DELETE_SCENARIO", "False") == "True"
    # -> write GIS information for grid
    WRITE_GRID_TO_GIS: bool = os.getenv("WRITE_GRID_TO_GIS", "False") == "True"
    # -> write electric vehicle data
    WRITE_EV: bool = os.getenv("WRITE_EV", "True") == "True"
    # -> write consumer summary
    WRITE_CONSUMER_SUMMARY: bool = os.getenv("WRITE_CONSUMER_SUMMARY", "True") == "True"

    # https://stackoverflow.com/questions/69090253/how-to-iterate-over-attributes-of-dataclass-in-python
    def get_config_dict(self):
        config = {}
        for field in self.__dataclass_fields__:
            config[str(field).lower()] = getattr(self, field)
        return config
