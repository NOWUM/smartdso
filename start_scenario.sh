#!/bin/bash

for i in "10.13.10.54 b2" "10.13.10.55 a1" "10.13.10.56 a2" "10.13.10.58 a3"; do 
    a=( $i ); 
    server="${a[0]}"
    case="${a[1]}"
    echo "$server ";
    echo "$case";

    python scenario_generator.py --case $case --slp
    scp ./docker-compose.yml nowum@$server:~/smartdso/
    ssh nowum@$server "docker system prune -f"
    ssh nowum@$server "docker volume prune -f"
    ssh nowum@$server "docker pull registry.git.fh-aachen.de/nowum-energy/projects/smartdso/smartdso:latest"
    ssh nowum@$server "cd smartdso && docker-compose down --remove-orphans; docker-compose up -d"
done
