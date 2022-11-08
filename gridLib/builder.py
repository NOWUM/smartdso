from gridLib.converter.cgmes_converter import CGMESConverter
from gridLib.converter.pv_potential import PVConverter
from gridLib.converter.heat_demand import HeatConverter
from gridLib.model import GridModel
import pandas as pd
import numpy as np

LONDON_IDS = np.load(r'./gridLib/data/london_ids.npy', allow_pickle=True)

PV_PATH = r"./gridLib/data/pv_potential/heinsberg/Solarkataster-Potentiale_05370016_Heinsberg_EPSG25832_Shape.shp"
PV_CONVERTER = PVConverter(PV_PATH)

HEAT_PATH = r"./gridLib/data/heat_demand/heinsberg"
HEAT_CONVERTER = HeatConverter(HEAT_PATH)

# GRID_PATH = r"./gridLib/data/import/dem/Export.xml"
# GRID_CONVERTER = CGMESConverter(path=GRID_PATH, levels=())
# GRID_CONVERTER.convert()
# CONSUMERS = GRID_CONVERTER.components["consumers"]

CONSUMERS = pd.read_csv(r'gridLib/data/export/alliander/consumers.csv', index_col=0)

# DEMAND_PATH = r"./gridLib/data/import/alliander/consumer_data.xlsx'"
#
# if DEMAND_PATH is not None:
#     demand = pd.read_excel(DEMAND_PATH, sheet_name="demand")
#     demand = demand.set_index("id_")
#     demand.columns = ["jeb"]
#     # -> set demand for each consumer
#     idx = CONSUMERS['id_'].isin(demand.index)
#     CONSUMERS = CONSUMERS.loc[idx]
#     CONSUMERS = CONSUMERS.join(demand, on="id_")
#     CONSUMERS = CONSUMERS.dropna()
# else:
#     CONSUMERS['jeb'] = 4500

CONSUMERS.rename(columns={'jeb': 'demand_power'}, inplace=True)
CONSUMERS['pv'] = None
CONSUMERS['demand_heat'] = 10_000

#CONSUMERS['london_data'] = [np.random.choice(LONDON_IDS) for _ in range(len(CONSUMERS))]
#CONSUMERS['pv'] = PV_CONVERTER.add_pv_to_consumers(CONSUMERS.copy())

#years, demand = HEAT_CONVERTER.add_heat_demand_to_consumers(CONSUMERS.copy())
#CONSUMERS['year'] = years
#CONSUMERS['demand_heat'] = demand

nodes = pd.read_csv(r'gridLib/data/export/alliander/nodes.csv', index_col=0)
lines = pd.read_csv(r'gridLib/data/export/alliander/lines.csv', index_col=0)
transformers = pd.read_csv(r'gridLib/data/export/alliander/transformers.csv', index_col=0)
MODEL = GridModel(
    nodes=nodes,
    lines=lines,
    transformers=transformers,
    consumers=CONSUMERS
)


def get_sub_grid(consumer_node):
    return int(MODEL.model.buses.loc[consumer_node, 'sub_network'])


CONSUMERS['sub_grid'] = CONSUMERS['bus0'].apply(get_sub_grid)
#
# GRID_CONVERTER.save(r"./gridLib/data/export/dem")