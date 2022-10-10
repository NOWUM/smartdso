# SmartDSO


## Documentation

The input set is as follows:

* london_data - if actual data from london measurements should be used (see [here](https://data.london.gov.uk/dataset/smartmeter-energy-use-data-in-london-households)); otherwise, standard load profiles are used
* ev_ratio - how many of the deployed cars are of type ev, other cars will be simulated but do not consume electricity
* pv_ratio - how many houses, which can have a PV have a PV in the simulation
* number_consumers - option to limit the simulated consumers (0 means unlimited)
* scenario - one of:
    1. PlugInCap
    2. MaxPvCap
    3. MaxPvSoc
    4. PlugInInf
* scenario_name -> if it contains Flat, Flat prices are used, else Spot -> electricity price only
    * Flat: single average of the Spot Price from the simulation interval
    * Spot: hourly spot price from simulation interval
* iteration - use the iteration index as a random seed to get different reproducible runs
* sub_grid - calculates a different part of the grid -> used for parallelization of calculating the different grids
* database_uri - URI of the database where the results are stored

The grid fee curve is used to calculate the grid fees which are added to the electricity price to build the final consumer price.
The consumers then decide wether they need to charge, based on the current SoC (state of charge).

## Scientific Paper - RMIT

* Different Pv Scenarios are evaluated:
    1. tariff Flat+PlugInCap -> Case 1 - Status Quo
    2. tariff Flat+MaxPvCap -> Case 2 - own consumption optimization
    3. tariff Spot+MaxPvCap -> Case 3 - own consumption optimization with marktsignal
    4. tariff Spot+MaxPvSoc -> Case 4 - Nutzenfunktion auf Basis des Füllstandsniveaus

## Simulation

1. to start the simulation with docker run the command:
`sudo chmod -R o+rw .` in sim_result directory.
2. run `python scenario_generator.py --case b --slp` to generate the docker-compose.yml
3. `docker-compose up -d`