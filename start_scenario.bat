:: #!/bin/sh
python compose_generator.py --pv 50
ssh nowum@10.13.10.56 "docker pull registry.git.fh-aachen.de/nowum-energy/projects/smartdso/smartdso:latest"
scp .\docker-compose.yml nowum@10.13.10.56:~/smartdso/
ssh nowum@10.13.10.56 "cd smartdso && docker-compose down --remove-orphans; docker-compose up -d"