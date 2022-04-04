from itertools import product
image_repo = 'registry.git.fh-aachen.de/nowum-energy/projects/smartdso/'

output = []
output.append('version: "3.9"\n')
output.append('services:\n')

ev_quotas = [100]
minimum_socs = [30]
number_simulation = 30

start_date = '2022/01/01'
end_date = '2022/01/15'

use_london_data = False

for scenario in product(ev_quotas, minimum_socs):
    for simulation in range(number_simulation):
        output.append(f'''
          scenario_{scenario[0]}_{scenario[1]}_{simulation}:
            container_name: s{scenario[0]}{scenario[1]}_{simulation}
            image: {image_repo}smartdso:latest
            build: .
            environment:
              EV_RATIO: {scenario[0]}
              MINIMUM_SOC: {scenario[1]}
              BASE_PRICE: 29
              LONDON_DATA: {use_london_data}
              START_DATE: {start_date}
              END_DATE: {end_date}
              SCENARIO_NAME: EV{scenario[0]}LIMIT{scenario[1]}_{simulation}
              RESULT_PATH: EV{scenario[0]}LIMIT{scenario[1]}
            volumes:
              - ./sim_result:/src/sim_result
        ''')

with open('docker-compose.yml', 'w') as f:
    f.writelines(output)