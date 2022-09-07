:: #!/bin/sh
set server=10.13.10.58
set slope=4.0
set strategy=optimized
set pv=80
:: python utils.py
:: python ./agents/capacity_provider.py
python compose_generator.py --pv %pv% --prc_sense %slope% --strategy %strategy%
ssh nowum@%server% "docker pull registry.git.fh-aachen.de/nowum-energy/projects/smartdso/smartdso:latest"
scp .\docker-compose.yml nowum@%server%:~/smartdso/
ssh nowum@%server% "cd smartdso && docker-compose down --remove-orphans; docker-compose up -d"
ssh nowum@%server%