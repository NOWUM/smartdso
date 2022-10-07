#!/bin/bash

for i in "10.13.10.54 a" "10.13.10.55 b" "10.13.10.56 c" "10.13.10.58 d"; do 
    a=( $i ); 
    server="${a[0]}"
    ssh nowum@$server "cd smartdso && docker-compose down --remove-orphans"
done