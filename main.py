import logging
import secrets
import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import timedelta as td

from agents.flexibility_provider import FlexibilityProvider
from agents.capacity_provider import CapacityProvider
from agents.utils import WeatherGenerator

from participants.basic import BasicParticipant, DataType
from participants.business import BusinessModel
from participants.industry import IndustryModel
from participants.residential import HouseholdModel

from utils import TableCreator

# -> timescaledb connection to store the simulation results
from config import SimulationConfig as Config
logging.basicConfig()
logging.getLogger("smartdso.residential").setLevel("WARNING")
logger = logging.getLogger("smartdso")
logger.setLevel("INFO")

config_dict = Config().get_config_dict()

try:
    tables = TableCreator(create_tables=Config.RESET_DATABASE, database_uri=Config.DATABASE_URI)
    logger.info(f" -> connected to database")
    logger.info(f" -> deleting scenario: {Config.NAME}")
    if Config.DELETE_SCENARIO:
        tables.delete_scenario(scenario=Config.NAME)
    logger.info(f" -> deleted scenario: {Config.NAME}")
except Exception as e:
    logger.error(f" -> can't connect to {Config.DATABASE_URI}")
    logger.error(repr(e))
    engine = None
    raise Exception("Bad simulation parameters, please check your input")


logger.info(f" -> initialize simulation for {Config.START_DATE.date()} - {Config.END_DATE.date()}")
# -> simulation time range and steps per day
time_range = pd.date_range(start=Config.START_DATE, end=Config.END_DATE + td(days=1), freq=Config.RESOLUTION)[:-1]
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
    idx = weather_at_each_day.index.isin(list(time_range))
    weather_at_each_day = weather_at_each_day.loc[idx]
    logger.info(" -> weather data simulation completed")

    # -> start capacity provider
    logger.info(" -> starting Capacity Provider")
    CapProvider = CapacityProvider(**config_dict)
    grid_fee = CapProvider.get_grid_fee(time_range=time_range)
    logger.info(" -> started Capacity Provider")
    logging.getLogger("smartdso.capacity_provider").setLevel("DEBUG")

    # -> start flexibility provider
    logger.info(" -> starting Flexibility Provider")
    FlexProvider = FlexibilityProvider(random=random, **config_dict)
    tariff = FlexProvider.get_tariff(time_range=time_range)
    logger.info(" -> started Flexibility Provider")
    logging.getLogger("smartdso.flexibility_provider").setLevel("DEBUG")
    config_dict['tariff'] = tariff
    # -> start consumer agents
    consumer_path = rf"./gridLib/data/export/{Config.GRID_DATA}/consumers.csv"
    consumers = pd.read_csv(consumer_path, index_col=0)
    fixed_demand = []
    fixed_generation = []
    if Config.SUB_GRID >= 0:
        idx = consumers['sub_grid'] == Config.SUB_GRID
        consumers = consumers.loc[idx]
    logger.info(f" -> starting {len(consumers)} clients in sub grid {Config.SUB_GRID}")
    clients: dict[str, BasicParticipant] = {}
    # -> all residential consumers
    residential_consumers = consumers.loc[consumers["profile"] == "H0"]
    residential_consumers = residential_consumers.fillna('[]')
    logger.info(f" -> starting residential clients")
    num_ = len(residential_consumers)
    for _, consumer in tqdm(residential_consumers.iterrows(), total=num_):
        # -> create unique identifier for each consumer
        id_ = secrets.token_urlsafe(8)
        # -> check pv potential and add system corresponding to the pv ratio
        pv_systems = []
        if random.choice(a=[True, False], p=[Config.PV_RATIO, 1 - Config.PV_RATIO]):
            pv_systems = eval(consumer["pv"])
        if random.choice(a=[True, False], p=[Config.HP_RATIO, 1 - Config.HP_RATIO]):
            hp_analyse = True
        else:
            hp_analyse = False

        clients[id_] = HouseholdModel(
            demand_power=consumer['demand_power'],
            demand_heat=consumer['demand_heat'],
            consumer_id=id_,
            grid_node=consumer["bus0"],
            residents=int(max(consumer["demand_power"] / 1500, 1)),
            london_id=consumer["london_data"],
            pv_systems=pv_systems,
            consumer_type="household",
            hp_analyse=hp_analyse,
            random=random,
            weather=weather_at_each_day,
            grid_fee=grid_fee,
            **config_dict
        )
        FlexProvider.register(id_, clients[id_])

    logger.info(f" -> started {len(residential_consumers)} residential clients")

    # -> all business consumers
    business_consumers = consumers.loc[consumers["profile"] == "G0"]
    logger.info(f" -> starting business clients")
    num_ = len(business_consumers)
    for _, consumer in tqdm(business_consumers.iterrows(), total=num_):
        # -> create unique identifier for each consumer
        id_ = secrets.token_urlsafe(8)
        clients[id_] = BusinessModel(
            demandP=consumer['demand_power'],
            consumer_id=id_,
            consumer_type="business",
            weather=weather_at_each_day,
            grid_fee=grid_fee,
            **config_dict
        )
        FlexProvider.register(id_, clients[id_])

    logger.info(f" -> started {len(business_consumers)} business clients")

    # -> all industry consumers
    industry_consumers = consumers.loc[consumers["profile"] == "RLM"]
    logger.info(f" -> starting industry clients")
    num_ = len(industry_consumers)
    for _, consumer in tqdm(industry_consumers.iterrows(), total=num_):
        # -> create unique identifier for each consumer
        id_ = secrets.token_urlsafe(8)
        clients[id_] = IndustryModel(
            demandP=consumer['demand_power'],
            consumer_id=id_,
            consumer_type="industry",
            weather=weather_at_each_day,
            grid_fee=grid_fee,
            **config_dict
        )
        FlexProvider.register(id_, clients[id_])
    logger.info(f" -> started {len(industry_consumers)} industry clients")


except Exception as e:
    logger.error(f" -> can't initialize agents")
    logger.error(repr(e))
    raise Exception(f"Bad simulation parameters, please check your input {repr(e)}")


if __name__ == "__main__":

    try:
        # -> collect demand for each day in simulation horizon
        logger.info(f" -> running photovoltaic and slp generation")
        fixed_demand = []
        for client in clients.values():
            fixed_demand.append(client.get(DataType.residual_demand, build_dataframe=True))
        # -> forward demand data to the Capacity Provider
        logger.info(f" -> running initial power flow calculation")
        CapProvider.set_fixed_power(data=pd.concat(fixed_demand))
    except Exception as e:
        print(repr(e))
        logger.error(f" -> error in power flow calculation: {repr(e)}")

    # -> start simulation for date range start_date till end_date
    for day in date_range[:-1]:
        logger.info(f" -> running day {day.date()}")
        number_clients = len(FlexProvider.keys)
        for d_time in time_range[time_range.date == day]:
            try:
                logger.info(f" -> consumers plan charging...")
                logger.info(f" -> {d_time}")
                number_commits = 0
                number_rejects = 0
                requested = {}
                while number_commits < number_clients:
                    for request, node_id in FlexProvider.get_requests(d_time=d_time):
                        price = CapProvider.handle_request(request=request, node_id=node_id)
                        if FlexProvider.commit(price):
                            CapProvider.set_demand(request=request, node_id=node_id)
                        else:
                            id_ = FlexProvider.consumer_handler.id_
                            if id_ in requested.keys():
                                if all(requested[id_] == request.values):
                                    print(f'same request of id:  {id_}')
                                else:
                                    print(f'get new request of id: {id_}')
                            requested[id_] = request.values
                            # print(f'reject: {id_}')
                            number_rejects += 1
                    if 'optimize' in Config.STRATEGY:
                        number_commits = FlexProvider.get_commits()
                        logger.debug(f" -> {number_commits} consumers commit charging")
                    elif 'heuristic' in Config.STRATEGY:
                        logger.debug("set commit charging for clients")
                        number_commits = len(clients)
                    # print(number_rejects)
                    number_rejects = 0
                for client in clients.values():
                    client.simulate(d_time)

            except Exception as e:
                logger.exception(f" -> error during simulation: {repr(e)}")

        if Config.WRITE_EV:
            for client in clients.values():
                client.save_ev_data(day)
                client.save_hp_data(day)
        if Config.WRITE_CONSUMER_SUMMARY:
            FlexProvider.save_consumer_summary(day)

        CapProvider.save_results(day)