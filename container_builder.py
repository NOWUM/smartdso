from itertools import product
image_repo = 'registry.git.fh-aachen.de/nowum-energy/projects/smartdso/'

output = []
output.append('version: "3.9"\n')
output.append('services:\n')

ev_quotas = [50, 80, 100]
minimum_socs = [30, 50, 80]

start_date = '2022/01/01'
end_date = '2022/02/01'

use_london_data = False

for scenario in product(ev_quotas, minimum_socs):
    for simulation in range(5):
        output.append(f'''
          scenario_{scenario[0]}_{scenario[1]}_{simulation}:
            container_name: s_{scenario[0]}_{scenario[1]}_{simulation}
            image: {image_repo}smartdso:latest
            build: .
            environment:
              EV_RATIO: {scenario[0]}
              MINIMUM_SOC: {scenario[1]}
              LONDON_DATA: {use_london_data}
              START_DATE: {start_date}
              END_DATE: {end_date}
            volumes:
              - ./sim_result/{scenario[0]}_{scenario[1]}/{simulation}:/src/sim_result
        ''')

with open('docker-compose.yml', 'w') as f:
    f.writelines(output)