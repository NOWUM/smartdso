#!/bin/sh

python scenario_generator.py --case b3 --slp

server=10.13.10.58
scp .\docker-compose.yml nowum@$server:~/smartdso/
ssh nowum@$server "docker system prune -f"
ssh nowum@$server "docker volume prune -f"
ssh nowum@$server "docker pull registry.git.fh-aachen.de/nowum-energy/projects/smartdso/smartdso:latest"
ssh nowum@$server "cd smartdso && docker-compose down --remove-orphans; docker-compose up -d"
