import pandas as pd
import glob as gb
import numpy as np

path = r'./sim_result/test_result'

if __name__ == "__main__":

    files = gb.glob(fr'{path}/*.csv')

    distances = []
    on_tour = []
    socs = []

    for file in files:
        if 'resample' not in file:
            df = pd.read_csv(file, sep=';', decimal=',', index_col=0)
            distance = df['ref_distance'].values
            distances += [distance]
            x = np.asarray([0] + list(np.diff(distance)))
            x[x > 0] = 1
            on_tour += [x]
            soc = df['ref_soc'].values
            socs += [soc]

    for k in range(30):
        car = pd.DataFrame(dict(soc=socs[k], distance=distances[k], in_use=on_tour[k]), index=df.index)
        car.to_csv(fr'{path}/car_{k}.csv', sep=';', decimal=',')

