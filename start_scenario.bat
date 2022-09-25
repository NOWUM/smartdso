:: #!/bin/sh
set server=10.13.10.58
set slope=4.0
set strategy=PlugInCap
set pv=50
set tariff=Flat
set consumers=0
set num_offset=0
set num=1
set start=2022-03-01
set end=2022-03-31
:: tariff can be spot or flat
:: python utils.py
:: python ./agents/capacity_provider.py
:: python utils.py
python compose_generator.py --pv %pv% --prc_sense %slope% --strategy %strategy% --tariff %tariff% --consumers %consumers% --start %start% --end %end% --num %num% --num_offset %num_offset%
ssh nowum@%server% "docker system prune -f"
ssh nowum@%server% "docker pull registry.git.fh-aachen.de/nowum-energy/projects/smartdso/smartdso:latest"
scp .\docker-compose.yml nowum@%server%:~/smartdso/
ssh nowum@%server% "cd smartdso && docker-compose down --remove-orphans; docker-compose up -d"
ssh nowum@%server%
