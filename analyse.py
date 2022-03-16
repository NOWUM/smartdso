import pandas as pd
import numpy as np
from glob import glob as gb

path = r'./sim_result/S8030'


def get_data():
    data = []
    files = gb(pathname=fr'{path}/*.csv')
    for file in files:
        if 'resample' not in file:
            d = pd.read_csv(file, sep=';', decimal=',', index_col=0)
            d.index = pd.to_datetime(d.index)
            data += [d]
    return data


def analyse(attribute: str, dataframes: list):
    data = np.asarray([df[attribute].values.flatten() for df in dataframes]).T
    df = pd.DataFrame(data)
    df['average'] = np.mean(df, axis=1)
    df['minimum'] = np.min(df, axis=1)
    df['maximum'] = np.max(df, axis=1)
    df.index = dataframes[0].index
    if parameter not in ['commits', 'requests', 'rejects']:
        df = df.resample('15min').mean()
    else:
        df = df.resample('15min').sum()
    return df


if __name__ == "__main__":
    dfs = get_data()
    parameters = dfs[0].columns
    results = {}
    for parameter in parameters:
        results[parameter] = analyse(parameter, dfs)
        results[parameter].to_csv(fr'./sim_result/R8030/{parameter}.csv', sep=';', decimal=',')

