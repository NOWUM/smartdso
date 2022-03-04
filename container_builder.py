from itertools import product
image_repo = 'registry.git.fh-aachen.de/nowum-energy/projects/smartdso/'

output = []
output.append('version: "3.9"\n')
output.append('services:\n')

ev_quotas = [50]
minimum_socs = [30]

use_london_data = False

for scenario in product(ev_quotas, minimum_socs):
    output.append(f'''
      scenario_{scenario[0]}_{scenario[1]}:
        container_name: s_{scenario[0]}_{scenario[1]}
        image: {image_repo}smartdso:latest
        environment:
          EV_RATIO: {scenario[0]}
          MINIMUM_SOC: {scenario[1]}
          LONDON_DATA: {use_london_data}
        volumes:
          - ./sim_result/{scenario[0]}_{scenario[1]}:/home/admin/src/sim_result
    ''')

with open('docker-compose.yml', 'w') as f:
    f.writelines(output)