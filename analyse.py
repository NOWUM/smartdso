import pandas as pd
import numpy as np
from glob import glob as gb
from pathlib import Path
import os

scenario = 'EV80LIMIT30'
sim_path = fr'./sim_result/S_{scenario}'
result_path = fr'./sim_result/R_{scenario}'

if not Path(fr'{result_path}').is_dir():
    os.mkdir(fr'{result_path}')
if not Path(fr'{result_path}/csv'):
    os.mkdir(fr'{result_path}/csv')


def get_data():
    data = []
    files = gb(pathname=fr'{sim_path}/*.csv')
    for file in files:
        # if 'resampled' not in file and 'lmp' not in file:
        if 'result_1min_' in file:
            d = pd.read_csv(file, sep=';', decimal=',', index_col=0)
            d.index = pd.to_datetime(d.index)
            data += [d]
    return data


def analyse(attribute: str, dataframes: list):
    data = np.asarray([df[attribute].values.flatten() for df in dataframes]).T
    df = pd.DataFrame(data)
    df['Average'] = np.mean(df, axis=1)
    df['Minimum'] = np.min(df, axis=1)
    df['Maximum'] = np.max(df, axis=1)
    df.index = dataframes[0].index
    if attribute not in ['commits', 'requests', 'rejects']:
        df = df.resample('15min').mean()
    else:
        df = df.resample('15min').sum()
    columns = list(df.columns)
    df = df.loc[:, columns[-3:] + columns[:-3]]
    return df


def run():
    dfs = get_data()
    parameters = dfs[0].columns
    results = {}
    for parameter in parameters:
        results[parameter] = analyse(parameter, dfs)
        results[parameter].to_csv(fr'{result_path}/csv/{parameter}.csv', sep=';', decimal=',')
        if parameter in ['charged', 'soc', 'waiting', 'price', 'requests']:
            with pd.ExcelWriter(fr'{result_path}/{parameter}.xlsx', if_sheet_exists='replace', mode='a') as writer:
                results[parameter].to_excel(writer, sheet_name=parameter)


if __name__ == "__main__":
    run()

