import argparse

IMAGE_REPO = 'registry.git.fh-aachen.de/nowum-energy/projects/smartdso/'
EV_RATIO = 100
SIMULATIONS, GRIDS = 10, 10
START_DATE = '2022-05-01'
END_DATE = '2022-05-31'


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--case', type=str, required=False, default="b", help='Scenario')
    parser.add_argument('--slp', action=argparse.BooleanOptionalAction, help='Use SLP or London Data')
    return parser.parse_args()


PARAS = parse_args()
SLP = PARAS.slp


def build_compose_file(strategy: str, prices: str, pv_ratio: int, ev_ratio: int = EV_RATIO,
                       start_date: str = START_DATE, end_date: str = END_DATE, simulations: int = SIMULATIONS,
                       grids: int = GRIDS, slp: bool = SLP):

    insert = '' if slp else '-L'

    output = ['version: "3.9"\n', 'services:\n']
    for simulation in range(simulations):
        for sub_grid in range(grids):
            output.append(f'''
              s_{strategy}_{sub_grid}_{simulation}:
                container_name: {strategy}_{sub_grid}_{simulation}
                image: {IMAGE_REPO}smartdso:latest
                build: .
                environment:
                  LONDON_DATA: "{not slp}"
                  EV_RATIO: {ev_ratio}
                  PV_RATIO: {pv_ratio}
                  START_DATE: {start_date}
                  END_DATE: {end_date}
                  STRATEGY: {strategy}
                  RANDOM_SEED: {simulation + sub_grid}
                  ANALYSE_GRID: "True"
                  SUB_GRID: {sub_grid}
                  SCENARIO_NAME: {strategy}-PV{pv_ratio}-Price{prices}{insert}_{simulation}
            ''')

    return output


CASES = {
    "a": build_compose_file(strategy='PlugInCap', prices='Flat', pv_ratio=25),
    "a1": build_compose_file(strategy='PlugInCap', prices='Flat', pv_ratio=50),
    "a2": build_compose_file(strategy='PlugInCap', prices='Flat', pv_ratio=80),
    "a3": build_compose_file(strategy='PlugInCap', prices='Flat', pv_ratio=100),
    "b": build_compose_file(strategy='MaxPvCap', prices='Flat', pv_ratio=25),
    "b1": build_compose_file(strategy='MaxPvCap', prices='Flat', pv_ratio=50),
    "b2": build_compose_file(strategy='MaxPvCap', prices='Flat', pv_ratio=80),
    "b3": build_compose_file(strategy='MaxPvCap', prices='Flat', pv_ratio=100),
    "c": build_compose_file(strategy='MaxPvCap', prices='Spot', pv_ratio=80),
    "d": build_compose_file(strategy='MaxPvSoc', prices='Spot', pv_ratio=80),
}


if __name__ == "__main__":

    case = CASES[PARAS.case]
    with open(f'docker-compose.yml', 'w') as f:
        f.writelines(case)