import logging
import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import timedelta as td

from agents.capacity_provider import CapacityProvider
from agents.flexibility_provider import FlexibilityProvider
from agents.utils import WeatherGenerator

from participants.basic import BasicParticipant, DataType
from participants.business import BusinessModel
from participants.industry import IndustryModel
from participants.residential import HouseholdModel

from utils import TableCreator
import uuid

# -> timescaledb connection to store the simulation results
from config import SimulationConfig as Config
logging.basicConfig()
logger = logging.getLogger("smartdso")
logger.setLevel("INFO")

config_dict = Config().get_config_dict()

try:
    tables = TableCreator(create_tables=Config.RESET_DATABASE, database_uri=Config.DATABASE_URI)
    logger.info(" -> connected to database")
except Exception as e:
    logger.error(f" -> can't connect to {Config.DATABASE_URI}")
    logger.error(repr(e))
    engine = None
    raise Exception("Bad simulation parameters, please check your input")


logger.info(f" -> initialize simulation for {Config.START_DATE.date()} - {Config.END_DATE.date()}")
# -> simulation time range and steps per day
time_range = pd.date_range(start=Config.START_DATE, end=Config.END_DATE + td(days=1), freq=Config.RESOLUTION)[-1]
date_range = pd.date_range(start=Config.START_DATE, end=Config.END_DATE + td(days=1), freq="d")

logger.info(f" -> and scenario {Config.NAME} with simulation no.: {Config.SIM} ")
logger.info(f" -> setting random seed to {Config.SEED}")
random = np.random.default_rng(Config.SEED)


try:
    # -> start weather generator
    logger.info(" -> starting Weather Generator and simulate weather data")
    WeatherGenerator = WeatherGenerator()
    # -> get weather data
    weather_at_each_day = []
    for day in date_range:
        weather_at_each_day.append(WeatherGenerator.get_weather(day))
    weather_at_each_day = pd.concat(weather_at_each_day)
    # -> resample to quarter hour values
    weather_at_each_day = weather_at_each_day.resample("15min").ffill()
    # -> remove not used time steps
    idx = weather_at_each_day.index.isin(time_range)
    weather_at_each_day = weather_at_each_day.loc[idx]
    logger.info(" -> weather data simulation completed")

    # -> start capacity provider
    logger.info(" -> starting Capacity Provider")
    # CapProvider = CapacityProvider(**config_dict)
    # grid_fee = CapProvider.get_grid_fee()
    logger.info(" -> started Capacity Provider")
    logging.getLogger("smartdso.capacity_provider").setLevel("DEBUG")

    # -> start flexibility provider
    logger.info(" -> starting Flexibility Provider")
    FlexProvider = FlexibilityProvider(**config_dict)
    tariff = FlexProvider.get_tariff()
    logger.info(" -> started Flexibility Provider")
    logging.getLogger("smartdso.flexibility_provider").setLevel("DEBUG")

    # -> start consumer agents
    consumer_path = rf"./gridLib/data/export/{Config.GRID_DATA}/consumers.csv"
    consumers = pd.read_csv(consumer_path, index_col=0)
    fixed_demand = []
    fixed_generation = []
    if Config.SUB_GRID >= 0:
        idx = consumers['sub_grid'] == Config.SUB_GRID
        consumers = consumers.loc[idx]
    logger.info(f" -> starting {len(consumers)} clients in sub grid {Config.SUB_GRID}")
    clients: dict[uuid.UUID, BasicParticipant] = {}
    # -> all residential consumers
    residential_consumers = consumers.loc[consumers["profile"] == "H0"]
    logger.info(f" -> starting residential clients")
    for _, consumer in residential_consumers.iterrows():
        # -> create unique identifier for each consumer
        id_ = uuid.uuid1()
        # -> check pv potential and add system corresponding to the pv ratio
        pv_systems = []
        if random.choice(a=[True, False], p=[Config.PV_RATIO, 1 - Config.PV_RATIO]):
            pv_systems = eval(consumer["pv"])

        clients[id_] = HouseholdModel(
            demand_power=consumer['demand_power'],
            demand_heat=consumer['demand_heat'],
            consumer_id=str(id_),
            grid_node=consumer["bus0"],
            residents=int(max(consumer["demand_power"] / 1500, 1)),
            london_id=consumer["london_data"],
            pv_systems=pv_systems,
            consumer_type="household",
            random=random,
            **config_dict
        )
        clients[id_].set_parameter(
            weather=weather_at_each_day.copy(),
            tariff=tariff.copy(),
            grid_fee=grid_fee.copy()
        )
        FlexProvider.register(id_, clients[id_])

    logger.info(f" -> started {len(residential_consumers)} residential clients")

    # -> all business consumers
    business_consumers = consumers.loc[consumers["profile"] == "G0"]
    logger.info(f" -> starting business clients")
    for _, consumer in business_consumers.iterrows():
        # -> create unique identifier for each consumer
        id_ = uuid.uuid1()
        clients[id_] = IndustryModel(
            demandP=consumer['demand_power'],
            consumer_id=str(id_),
            consumer_type="business",
            **config_dict
        )
        clients[id_].set_parameter(
            weather=weather_at_each_day.copy(),
            tariff=tariff.copy(),
            grid_fee=grid_fee.copy()
        )
        FlexProvider.register(id_, clients[id_])

    logger.info(f" -> started {len(business_consumers)} business clients")

    # -> all industry consumers
    industry_consumers = consumers.loc[consumers["profile"] == "RLM"]
    logger.info(f" -> starting industry clients")
    for _, consumer in industry_consumers.iterrows():
        # -> create unique identifier for each consumer
        id_ = uuid.uuid1()
        clients[id_] = BusinessModel(
            demandP=consumer['demand_power'],
            consumer_id=str(id_),
            consumer_type="industry",
            **config_dict
        )
        clients[id_].set_parameter(
            weather=weather_at_each_day.copy(),
            tariff=tariff.copy(),
            grid_fee=grid_fee.copy()
        )
        FlexProvider.register(id_, clients[id_])
    logger.info(f" -> started {len(industry_consumers)} industry clients")


except Exception as e:
    logger.error(f" -> can't initialize agents")
    logger.error(repr(e))
    raise Exception(f"Bad simulation parameters, please check your input {repr(e)}")


if __name__ == "__main__":
    pass

#     try:
#         # -> run SLPs for each day in simulation horizon
#         logger.info(f" -> running photovoltaic and slp generation")
#         fixed_demand = []
#         fixed_generation = []
#         for client in clients.values():
#             fixed_demand.append(client.get_initial_power(DataType.residual_demand))
#             fixed_generation.append(client.get_initial_power(DataType.residual_generation))
#         fixed_demand = pd.concat(fixed_demand)
#         fixed_generation = pd.concat(fixed_generation)
#
#         if analyse_grid:
#             # -> forward the slp data to the Capacity Provider
#             logger.info(f" -> running initial power flow calculation")
#             CapProvider.set_fixed_power(data=fixed_demand)
#     except Exception as e:
#         print(repr(e))
#         logger.error(f" -> error while slp or power flow calculation: {repr(e)}")
#
#     # -> start simulation for date range start_date till end_date
#     logger.info(" -> starting simulation")
#     for day in pd.date_range(start=start_date, end=end_date, freq="d"):
#         logger.info(f" -> running day {day.date()}")
#         logger.info(" -> consumers plan charging...")
#         try:
#             for d_time in tqdm(pd.date_range(start=day, periods=T, freq=RESOLUTION[T])):
#                 number_commits = 0
#                 while number_commits < len(FlexProvider.keys):
#                     for request, node_id, consumer_id in FlexProvider.get_requests(
#                         d_time=d_time, random=random
#                     ):
#                         if analyse_grid:
#                             price = CapProvider.get_price(
#                                 request=request, node_id=node_id
#                             )
#                             if FlexProvider.commit(price, consumer_id):
#                                 CapProvider.commit(request=request, node_id=node_id)
#                         else:
#                             price = pd.Series(
#                                 index=request.index, data=[0] * len(request)
#                             )
#                             FlexProvider.commit(price, consumer_id)
#
#                     if "MaxPv" in STRATEGY:
#                         number_commits = FlexProvider.get_commits()
#                         logger.debug(
#                             f" -> {FlexProvider.get_commits()} consumers commit charging"
#                         )
#                     elif "PlugIn" in STRATEGY:
#                         logger.debug("set commit charging for clients")
#                         number_commits = len(FlexProvider.keys)
#                 FlexProvider.simulate(d_time)
#             FlexProvider.save_results(day)
#             if analyse_grid:
#                 CapProvider.save_results(day)
#                 pass
#
#         except Exception as e:
#             logger.exception(f"Error during simulation")
