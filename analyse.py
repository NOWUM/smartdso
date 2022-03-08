import pandas as pd
import numpy as np
import sqlite3
from glob import glob as gb

path = r'./sim_result'


def get_data(data: pd.DataFrame, type_: str):
    return data[type_].values.flatten()


def save_data(data: list, columns: list, index: pd.Index, type_: str):
    x = pd.DataFrame(np.asarray(data).T, columns=columns, index=[pd.to_datetime(i) for i in index])
    if type_ == 'charged' or type_ == 'soc':
        x = x.resample('15min').mean()
    if type_ == 'requests':
        x = x.resample('15min').sum()

    average = x.mean(axis=1)
    max_ = x.max(axis=1)
    min_ = x.min(axis=1)
    x.loc[:, 'Average'] = average.values
    x.loc[:, 'Maximal'] = max_.values
    x.loc[:, 'Minimal'] = min_.values

    x.to_csv(fr'{path}/{type_}.csv', sep=';', decimal=',')


if __name__ == "__main__":

    databases = [file for file in gb(fr'{path}/*') if '.db' in file]
    df = pd.DataFrame()

    results = {type_: [] for type_ in ['power', 'charged', 'soc', 'requests', 'commits', 'price']}
    names = []

    for database in databases:
        db = sqlite3.connect(database)
        df = pd.read_sql('SELECT * FROM results', db)
        df = df.set_index('index')
        for key in results.keys():
            results[key] += [get_data(data=df, type_=key)]
        names.append(database.split('\\')[-1].replace('.db', ''))

    for key in results.keys():
        save_data(results[key], names, df.index, key)

