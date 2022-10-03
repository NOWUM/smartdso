import argparse

IMAGE_REPO = 'registry.git.fh-aachen.de/nowum-energy/projects/smartdso/'

# ------ Scenario ------
EV_RATIO = 100
PV_RATIO = 25
START_DATE = '2022-05-01'
END_DATE = '2022-05-31'
PRICES = 'Flat'
STRATEGY = 'PlugInCap'
NUM = 10
GRIDS = 10
LONDON = False

if __name__ == "__main__":

    output = ['version: "3.9"\n', 'services:\n']

    for simulation in range(NUM):
        for sub_gird in range(GRIDS):
            output.append(f'''
              s_{STRATEGY}_{sub_gird}_{simulation}:
                container_name: {STRATEGY}_{sub_gird}_{simulation}
                image: {IMAGE_REPO}smartdso:latest
                build: .
                environment:
                  LONDON_DATA: "{LONDON}"
                  EV_RATIO: {EV_RATIO}
                  PV_RATIO: {PV_RATIO}
                  START_DATE: {START_DATE}
                  END_DATE: {END_DATE}
                  STRATEGY: {STRATEGY}
                  RANDOM_SEED: {simulation + sub_gird}
                  ANALYSE_GRID: "True"
                  SUB_GRID: {sub_gird}
                  SCENARIO_NAME: {STRATEGY}-PV{PV_RATIO}-Price{PRICES}_{simulation}
            ''')

    with open(f'docker-compose.yml', 'w') as f:
        f.writelines(output)
