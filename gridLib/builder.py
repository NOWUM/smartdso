from gridLib.converter.cgmes_converter import CGMESConverter
from gridLib.converter.dwg_converter import DWGConverter
from gridLib.converter.pv_potential import PVConverter
from gridLib.converter.heat_demand import HeatConverter
from gridLib.converter.lat_lon_finder import LonLatFinder
from gridLib.model import GridModel
import pandas as pd
import numpy as np

LONDON_IDS = np.load(r'./gridLib/data/london_ids.npy', allow_pickle=True)


if __name__ == "__main__":

    area = 'heinsberg'

    if area == 'heinsberg':
        pv = r"./gridLib/data/pv_potential/heinsberg/Solarkataster-Potentiale_05370016_Heinsberg_EPSG25832_Shape.shp"
        heat = r"./gridLib/data/heat_demand/heinsberg"
        grid = r"./gridLib/data/import/alliander/Porselen_Daten_7.dxf"
        demand = r"./gridLib/data/import/alliander/consumer_data.xlsx"
    else:
        pv = r"gridLib/data/pv_potential/dueren/Solarkataster-Potentiale_05358008_Dueren_EPSG25832_Shape.shp"
        heat = r"./gridLib/data/heat_demand/dueren"
        grid = r"./gridLib/data/import/dem/Export.xml"
        demand = r"./gridLib/data/import/dem/consumer_data.xlsx"

    if 'dxf' in grid:
        grid_conv = DWGConverter(grid)
    else:
        grid_conv = CGMESConverter(grid)

    grid_conv.convert()

    model = GridModel(
        nodes=grid_conv.components['nodes'],
        lines=grid_conv.components['lines'],
        transformers=grid_conv.components['transformers'],
        consumers=grid_conv.components['consumers']
    )

    def get_sub_grid(consumer_node):
        return int(model.model.buses.loc[consumer_node, 'sub_network'])
    consumers = grid_conv.components['consumers']
    consumers['sub_grid'] = consumers['bus0'].apply(get_sub_grid)

    if demand is not None:
        demand = pd.read_excel(demand, sheet_name="demand")
        demand = demand.set_index("id_")
        # -> set demand for each consumer
        idx = consumers.index.isin(demand.index)
        if any(idx):
            consumers = consumers.loc[idx]
            consumers = consumers.join(demand, on="id_")
            consumers = consumers.dropna()
        else:
            consumers['profile'] = "H0"
            finder = LonLatFinder(demand)
            finder.find_coords()
            consumers = finder.map_to_grid(consumers)
    else:
        consumers['jeb'] = 4500

    consumers.rename(columns={'jeb': 'demand_power'}, inplace=True)

    pv_converter = PVConverter(pv)
    pv_consumers = pv_converter.add_pv_to_consumers(consumers.copy())

    for idx, data in pv_consumers.items():
        consumers.at[idx, 'pv'] = data

    heat_converter = HeatConverter(heat)
    years, demand = heat_converter.add_heat_demand_to_consumers(consumers.copy())
    consumers['year'] = years
    consumers['demand_heat'] = demand

    consumers['london_data'] = np.random.choice(LONDON_IDS, size=len(consumers))

    grid_conv.save(r"./gridLib/data/export/alliander/")