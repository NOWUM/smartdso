:: #!/bin/sh
set server=10.13.10.58
set slope=4.0
set strategy=PlugInInf
set pv=100
set tariff=Flat
set consumers=10
set num=200
set start=2022-01-01
set end=2022-04-30
:: tariff can be spot or flat
:: python utils.py
:: python ./agents/capacity_provider.py
python compose_generator.py --pv %pv% --prc_sense %slope% --strategy %strategy% --tariff %tariff% --consumers %consumers% --start %start% --end %end% --num %num%
::ssh nowum@%server% "docker pull registry.git.fh-aachen.de/nowum-energy/projects/smartdso/smartdso:latest"
::scp .\docker-compose.yml nowum@%server%:~/smartdso/
::ssh nowum@%server% "cd smartdso && docker-compose down --remove-orphans; docker-compose up -d"
::ssh nowum@%server%