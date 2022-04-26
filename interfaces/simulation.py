import logging
import paramiko
import os


logger = logging.getLogger('simulation_server')

api_key = 'pk.eyJ1Ijoicmlla2VjaCIsImEiOiJjazRiYTdndXkwYnN3M2xteGN2MHhtZjB0In0.33tSDK45TXF3lb3-G147jw'
pk = paramiko.Ed25519Key.from_private_key(open(r'C:\Users\rieke\.ssh\id_ed25519'))
image_repo = 'registry.git.fh-aachen.de/nowum-energy/projects/smartdso/'


def update_image(s):
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(s, 22, 'nowum', pkey=pk)
    command = f'docker pull {image_repo}smartdso:latest'
    stdin, stdout, stderr = ssh.exec_command(command)
    error = stderr.readlines()
    if len(error) == 0:
        logger.info(f'updated image on {s}')
    else:
        logger.error(error)
    ssh.close()


def initialize_scenario(s, ev_ratio=100, minimum_soc=-1, start_date='2022/01/01', end_date='2022/01/15'):
    output = ['version: "3.9"\n', 'services:\n']

    transport = paramiko.Transport((s, 22))
    transport.connect(username='nowum', pkey=pk)
    sftp = paramiko.SFTPClient.from_transport(transport)

    for simulation in range(30):
        output.append(f'''
          scenario_{ev_ratio}_{minimum_soc}_{simulation}:
            container_name: s{ev_ratio}{minimum_soc}_{simulation}
            image: {image_repo}smartdso:latest
            build: .
            environment:
              EV_RATIO: {ev_ratio}
              MINIMUM_SOC: {minimum_soc}
              START_DATE: {start_date}
              END_DATE: {end_date}
              SCENARIO_NAME: EV{ev_ratio}LIMIT{minimum_soc}_{simulation}
        ''')

    logger.info(f'created scenario with ev ratio {ev_ratio} % and charging strategy {minimum_soc}')
    with open(f'EV{ev_ratio}LIMIT{minimum_soc}.yml', 'w') as f:
        f.writelines(output)

    sftp.put(f'EV{ev_ratio}LIMIT{minimum_soc}.yml', f'smartdso/docker-compose.yml')
    logger.info(f'put scenario file on {s}')
    os.remove(f'EV{ev_ratio}LIMIT{minimum_soc}.yml')

    sftp.close()


def start_scenario(s):
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(s, 22, 'nowum', pkey=pk)
    command = 'cd smartdso && docker-compose up --remove-orphans -d'
    ssh.exec_command(command)
    logger.info(f'started scenario on {s}')

    ssh.close()

