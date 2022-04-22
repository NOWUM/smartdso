import os
import paramiko

pk = paramiko.Ed25519Key.from_private_key(open(r'C:\Users\rieke\.ssh\id_ed25519'))
image_repo = 'registry.git.fh-aachen.de/nowum-energy/projects/smartdso/'
servers = ["10.13.10.54", "10.13.10.55", "10.13.10.56"]


def update_image(server):
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(server, 22, 'nowum', pkey=pk)
    command = f'docker pull {image_repo}smartdso:latest'
    stdin, stdout, stderr = ssh.exec_command(command)
    error = stderr.readlines()
    if len(error) == 0:
        print(f'updated image on {server}')
    else:
        print(error)
    ssh.close()


def initialize_scenario(server, ev_ratio=100, minimum_soc=-1, num=30, start_date='2022/01/01', end_date='2022/01/15'):
    output = ['version: "3.9"\n', 'services:\n']

    transport = paramiko.Transport((server, 22))
    transport.connect(username='nowum', pkey=pk)
    sftp = paramiko.SFTPClient.from_transport(transport)

    for simulation in range(num):
        output.append(f'''
          scenario_{ev_ratio}_{minimum_soc}_{simulation}:
            container_name: s{ev_ratio}{minimum_soc}_{simulation}
            image: {image_repo}smartdso:latest
            build: .
            environment:
              EV_RATIO: {ev_ratio}
              MINIMUM_SOC: {minimum_soc}
              BASE_PRICE: 29
              START_DATE: {start_date}
              END_DATE: {end_date}
              SCENARIO_NAME: EV{ev_ratio}LIMIT{minimum_soc}_{simulation}
            volumes:
              - ./sim_result:/src/sim_result
        ''')

    print(f'created scenario with ev ratio {ev_ratio} % and charging strategy {minimum_soc}')
    with open(f'EV{ev_ratio}LIMIT{minimum_soc}.yml', 'w') as f:
        f.writelines(output)

    sftp.put(f'EV{ev_ratio}LIMIT{minimum_soc}.yml', f'smartdso/docker-compose.yml')
    print(f'put scenario file on {server}')
    os.remove(f'EV{ev_ratio}LIMIT{minimum_soc}.yml')

    sftp.close()


def start_scenario(server):
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(server, 22, 'nowum', pkey=pk)
    command = 'cd smartdso && docker-compose up --remove-orphans -d'
    stdin, stdout, stderr = ssh.exec_command(command)
    error = stderr.readlines()
    if len(error) == 0:
        print(f'started scenario on {server}')
    else:
        print(error)
    ssh.close()


if __name__ == "__main__":
    print('--> update image file')
    for s in servers:
        update_image(s)
    print(' --> initialize scenarios')
    initialize_scenario(server=servers[0], minimum_soc=-1)
    initialize_scenario(server=servers[1], minimum_soc=100)
    initialize_scenario(server=servers[2], minimum_soc=30)
    print(' --> start scenarios')
    for s in servers:
        start_scenario(s)
