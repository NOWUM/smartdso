import argparse

IMAGE_REPO = 'registry.git.fh-aachen.de/nowum-energy/projects/smartdso/'


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ev', type=int, required=False, default=100, help='EV-Ratio')
    parser.add_argument('--pv', type=int, required=False, default=100, help='PV-Ratio')
    parser.add_argument('--num', type=int, required=False, default=30, help='Number of Simulations')
    parser.add_argument('--london', type=bool, required=False, default=True, help='Use London Data')
    parser.add_argument('--start', type=str, required=False, default='2015-08-01', help='Start Date')
    parser.add_argument('--end', type=str, required=False, default='2015-08-31', help='End Date')
    return parser.parse_args()


if __name__ == "__main__":
    paras = parse_args()

    output = ['version: "3.9"\n', 'services:\n']

    for simulation in range(30):
        output.append(f'''
          scenario_{paras.ev}_{paras.pv}_{simulation}:
            container_name: s{paras.ev}{paras.pv}_{simulation}
            image: {IMAGE_REPO}smartdso:latest
            build: .
            environment:
              LONDON_DATA: "{paras.london}"
              EV_RATIO: {paras.ev}
              PV_RATIO: {paras.pv}
              START_DATE: {paras.start}
              END_DATE: {paras.end}
              SCENARIO_NAME: EV{paras.ev}PV{paras.pv}_{simulation}
        ''')

    with open(f'docker-compose.yml', 'w') as f:
        f.writelines(output)
