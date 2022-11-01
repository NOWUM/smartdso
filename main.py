import logging
import os

import pandas as pd
import numpy as np
from tqdm import tqdm

from agents.capacity_provider import CapacityProvider
from agents.flexibility_provider import FlexibilityProvider
from participants.basic import BasicParticipant
from utils import TableCreator
import uuid

logging.basicConfig()
logger = logging.getLogger("smartdso")
logger.setLevel("INFO")

# -> timescaledb connection to store the simulation results
from config import DATABASE_URI, SUB_GRID, RESOLUTION

try:
    tables = TableCreator(create_tables=False, database_uri=DATABASE_URI)
    logger.info(" -> connected to database")
except Exception as e:
    logger.error(f" -> can't connect to {DATABASE_URI}")
    logger.error(repr(e))
    engine = None
    raise Exception("Bad simulation parameters, please check your input")

start_date = pd.to_datetime(
    os.getenv("START_DATE", "2022-05-01")
)  # -> default start date
end_date = pd.to_datetime(os.getenv("END_DATE", "2022-05-10"))  # -> default end date
GRID_DATA = os.getenv("GRID_DATA", "dem")
SEED = int(os.getenv("RANDOM_SEED", 2022))
T = int(os.getenv("STEPS_PER_DAY", 96))
LONDON_DATA = os.getenv("LONDON_DATA", "False") == "True"
# -> Need london data set
NUMBER_CONSUMERS = int(os.getenv("NUMBER_CONSUMERS", 0))
EV_RATIO = int(os.getenv("EV_RATIO", 100)) / 100
PV_RATIO = int(os.getenv("PV_RATIO", 100)) / 100
logger.info(f" -> initialize simulation for {start_date.date()} - {end_date.date()}")

scenario_name = os.getenv("SCENARIO_NAME", "Test_0")
# -> PlugInCap, MaxPvCap, MaxPvSoc, PlugInInf
STRATEGY = os.getenv("STRATEGY", "PlugInCap")
sim = int(os.getenv("RESULT_PATH", scenario_name.split("_")[-1]))
analyse_grid = os.getenv("ANALYSE_GRID", "True") == "True"
random = np.random.default_rng(SEED)

logger.info(f' -> scenario {scenario_name.split("_")[0]} and iteration {sim}')

input_set = {
    "start_date": start_date,  #    see: demLib.london_data.py
    "end_date": end_date,
    "T": T,
    "scenario": scenario_name.split("_")[0],
    "iteration": sim,
    "strategy": STRATEGY,
    "sub_grid": SUB_GRID,
    "database_uri": DATABASE_URI,
}

from participants.business import BusinessModel
from participants.industry import IndustryModel
from participants.residential import HouseholdModel


def create_agents(
    grid_series,
    sub_grid,
    start_date,
    end_date,
    scenario,
    number_consumers,
    random,
    ev_ratio,
    pv_ratio,
    london_data,
) -> dict[uuid.UUID, BasicParticipant]:
    clients: dict[uuid.UUID, BasicParticipant] = {}
    # -> read known consumers and nodes
    consumers = pd.read_csv(
        rf"./gridLib/data/export/{GRID_DATA}/consumers.csv", index_col=0
    )

    consumers["sub_grid"] = [
        grid_series.loc[node] if node in grid_series.index else -1
        for node in consumers["bus0"].values
    ]
    consumers = consumers.loc[consumers["sub_grid"] != -1]

    if "profile" not in consumers.columns:
        consumers["profile"] = "H0"
    if "pv" not in consumers.columns:
        consumers["pv"] = 0
    if "jeb" not in consumers.columns:
        # Jahresenergieverbrauch
        consumers["jeb"] = 4500
    if "london_data" not in consumers.columns:
        consumers["london_data"] = "MAC001844"
    if sub_grid != -1:
        consumers = consumers.loc[consumers["sub_grid"] == str(sub_grid)]

    h0_consumers = consumers.loc[consumers["profile"] == "H0"]  # -> all h0 consumers
    h0_consumers = h0_consumers.fillna(0)  # -> without pv = 0
    g0_consumers = consumers.loc[consumers["profile"] == "G0"]  # -> all g0 consumers
    rlm_consumers = consumers.loc[consumers["profile"] == "RLM"]  # -> all rlm consumers

    logger.info(f" -> building {len(consumers)} consumers in sub grid {sub_grid}")

    if number_consumers > 0:
        h0_consumers = h0_consumers.sample(number_consumers)

    # -> create household clients
    for _, consumer in h0_consumers.iterrows():
        # -> check pv potential and add system corresponding to the pv ratio
        if consumer["pv"] == 0:
            pv_systems = []
        elif random.choice(a=[True, False], p=[pv_ratio, 1 - pv_ratio]):
            pv_systems = eval(consumer["pv"])
        else:
            pv_systems = []

        # -> initialize h0 consumers
        id_ = uuid.uuid1()
        client = HouseholdModel(
            demandP=consumer["jeb"],
            consumer_id=str(id_),
            grid_node=consumer["bus0"],
            residents=int(max(consumer["jeb"] / 1500, 1)),
            ev_ratio=ev_ratio,
            london_data=london_data,
            l_id=consumer["london_data"],
            pv_systems=pv_systems,
            random=random,
            strategy=STRATEGY,
            scenario=scenario,
            start_date=start_date,
            end_date=end_date,
            T=T,
            database_uri=DATABASE_URI,
            consumer_type="household",
        )

        clients[id_] = client

    # -> create business clients
    for _, consumer in g0_consumers.iterrows():
        client = BusinessModel(
            T=T,
            demandP=consumer["jeb"],
            grid_node=consumer["bus0"],
            start_date=start_date,
            end_date=end_date,
            consumer_type="business",
        )
        clients[uuid.uuid1()] = client

    # -> create industry clients
    for _, consumer in rlm_consumers.iterrows():
        client = IndustryModel(
            T=T,
            demandP=consumer["jeb"],
            grid_node=consumer["bus0"],
            start_date=start_date,
            end_date=end_date,
            consumer_type="industry",
        )
        clients[uuid.uuid1()] = client

    return clients


if "Test" in scenario_name:
    tables.delete_scenario(scenario=scenario_name.split("_")[0])

try:
    logger.info(" -> starting Capacity Provider")
    CapProvider = CapacityProvider(**input_set, write_geo=False)
    logger.info(" -> started Capacity Provider")
    logging.getLogger("smartdso.capacityprovider").setLevel("DEBUG")
    logger.info(" -> starting Flexibility Provider")
    clients = create_agents(
        CapProvider.mapper,
        SUB_GRID,
        start_date,
        end_date,
        scenario_name.split("_")[0],
        NUMBER_CONSUMERS,
        random,
        PV_RATIO,
        EV_RATIO,
        LONDON_DATA,
    )
    FlexProvider = FlexibilityProvider(clients=clients, **input_set)
    logger.info(" -> started Flexibility Provider")
    logging.getLogger("smartdso.flexibilityprovider").setLevel("DEBUG")
except Exception as e:
    logger.error(f" -> can't initialize agents")
    logger.error(repr(e))
    raise Exception(f"Bad simulation parameters, please check your input {repr(e)}")


if __name__ == "__main__":
    try:
        # -> run SLPs for each day in simulation horizon
        logger.info(f" -> running photovoltaic and slp generation")
        fixed_demand, fixed_generation = FlexProvider.initialize_time_series()
        if analyse_grid:
            # -> forward the slp data to the Capacity Provider
            logger.info(f" -> running initial power flow calculation")
            CapProvider.set_fixed_power(data=fixed_demand)
    except Exception as e:
        print(repr(e))
        logger.error(f" -> error while slp or power flow calculation: {repr(e)}")

    # -> start simulation for date range start_date till end_date
    logger.info(" -> starting simulation")
    for day in pd.date_range(start=start_date, end=end_date, freq="d"):
        logger.info(f" -> running day {day.date()}")
        logger.info(" -> consumers plan charging...")
        try:
            for d_time in tqdm(pd.date_range(start=day, periods=T, freq=RESOLUTION[T])):
                number_commits = 0
                while number_commits < len(FlexProvider.keys):
                    for request, node_id, consumer_id in FlexProvider.get_requests(
                        d_time=d_time, random=random
                    ):
                        if analyse_grid:
                            price = CapProvider.get_price(
                                request=request, node_id=node_id
                            )
                            if FlexProvider.commit(price, consumer_id):
                                CapProvider.commit(request=request, node_id=node_id)
                        else:
                            price = pd.Series(
                                index=request.index, data=[0] * len(request)
                            )
                            FlexProvider.commit(price, consumer_id)

                    if "MaxPv" in STRATEGY:
                        number_commits = FlexProvider.get_commits()
                        logger.debug(
                            f" -> {FlexProvider.get_commits()} consumers commit charging"
                        )
                    elif "PlugIn" in STRATEGY:
                        logger.debug("set commit charging for clients")
                        number_commits = len(FlexProvider.keys)
                FlexProvider.simulate(d_time)
            FlexProvider.save_results(day)
            if analyse_grid:
                CapProvider.save_results(day)
                pass

        except Exception as e:
            logger.exception(f"Error during simulation")
