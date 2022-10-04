:: #!/bin/sh
set server=10.13.10.55
:: steigung der E-Funktion -> jetzt Splines
set slope=4.0
:: Flat is mean EEX price
:: tariff Flat+PlugInCap -> Case 1 - Status Quo
:: tariff Flat+MaxPvCap -> Case 2 - own consumption opt
:: tariff Spot+MaxPvCap -> Case 3 - own consumption opt with marktsignal
:: tariff Spot+MaxPvSoc -> Case 4 - Nutzenfunktion auf Basis des Füllstandsniveaus
:: PlugInInf -> Tobis Case - infinite zahlungsbereitschaft
set strategy=MaxPvCap
set pv=80
:: Flat oder Spot
set tariff=Spot
:: 0 is infinite
set consumers=0
:: offset for consumer ids  
set num_offset=0
:: number of reproductions with different random seed
set num=10
:: resulting in num*subgrids containers -> 10*10
set start=2022-05-01
set end=2022-05-31
:: tariff can be spot or flat
:: python utils.py
:: python ./agents/capacity_provider.py
:: python utils.py
python compose_generator.py --pv %pv% --prc_sense %slope% --strategy %strategy% --tariff %tariff% --consumers %consumers% --start %start% --end %end% --num %num% --num_offset %num_offset%
ssh nowum@%server% "docker system prune -f"
ssh nowum@%server% "docker volume prune -f"
ssh nowum@%server% "docker pull registry.git.fh-aachen.de/nowum-energy/projects/smartdso/smartdso:latest"
scp .\docker-compose.yml nowum@%server%:~/smartdso/
ssh nowum@%server% "cd smartdso && docker-compose down --remove-orphans; docker-compose up -d"
