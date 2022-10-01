import argparse

IMAGE_REPO = 'registry.git.fh-aachen.de/nowum-energy/projects/smartdso/'

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ev', type=int, required=False, default=100, help='EV-Ratio')
    parser.add_argument('--pv', type=int, required=False, default=100, help='PV-Ratio')
    parser.add_argument('--num', type=int, required=False, default=30, help='Number of Simulations')
    parser.add_argument('--num_offset', type=int, required=False, default=1, help='offset for simulation number (id)')
    parser.add_argument('--tariff', type=str, required=False, default='const', help='Use variable or constant tariff')
    parser.add_argument('--london', type=bool, required=False, default=False, help='Use London Data')
    parser.add_argument('--start', type=str, required=False, default='2022-03-01', help='Start Date')
    parser.add_argument('--end', type=str, required=False, default='2022-03-31', help='End Date')
    parser.add_argument('--strategy', type=str, required=False, default='PlugInCap', help='charging strategy')
    parser.add_argument('--consumers', type=int, required=False, default=0, help='number of sample')

    return parser.parse_args()


if __name__ == "__main__":
    paras = parse_args()

    output = ['version: "3.9"\n', 'services:\n']

    strategy = f'{paras.strategy}'

    for simulation in range(paras.num_offset, paras.num_offset + paras.num):
        for sub_gird in range(10):
            output.append(f'''
              s_{paras.ev}{paras.pv}_{sub_gird}_{simulation}:
                container_name: s{paras.ev}{paras.pv}_{sub_gird}_{simulation}
                image: {IMAGE_REPO}smartdso:latest
                build: .
                environment:
                  LONDON_DATA: "{paras.london}"
                  EV_RATIO: {paras.ev}
                  PV_RATIO: {paras.pv}
                  START_DATE: {paras.start}
                  END_DATE: {paras.end}
                  STRATEGY: {paras.strategy}
                  RANDOM_SEED: {simulation + sub_gird}
                  ANALYSE_GRID: "True"
                  SUB_GRID: {sub_gird}
                  NUMBER_CONSUMERS: {paras.consumers}
                  SCENARIO_NAME: EV{paras.ev}PV{paras.pv}PRC{paras.tariff}STR{strategy}_{simulation}
            ''')

    with open(f'docker-compose.yml', 'w') as f:
        f.writelines(output)
