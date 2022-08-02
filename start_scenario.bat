:: #!/bin/sh
set server=10.13.10.55
set slope=4.0
:: python utils.py
:: python ./agents/capacity_provider.py
python compose_generator.py --pv 100 --prc_sense %slope%
ssh nowum@%server% "docker pull registry.git.fh-aachen.de/nowum-energy/projects/smartdso/smartdso:latest"
scp .\docker-compose.yml nowum@%server%:~/smartdso/
ssh nowum@%server% "cd smartdso && docker-compose down --remove-orphans; docker-compose up -d"
ssh nowum@%server%