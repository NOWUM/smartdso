import pandas as pd
import numpy as np
from glob import glob as gb
from pathlib import Path
import os

scenario = 'EV100LIMIT-1'
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
    if attribute not in ['commits', 'rejects']:
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
        results[parameter].to_csv(fr'{result_path}/{parameter}.csv', sep=';', decimal=',')
        #if parameter in ['charged', 'soc', 'waiting', 'price', 'requests']:
        #    with pd.ExcelWriter(fr'{result_path}/{parameter}.xlsx', if_sheet_exists='replace', mode='a') as writer:
        #        results[parameter].to_excel(writer, sheet_name=parameter)
    return results


if __name__ == "__main__":
    result = run()
    # df = pd.read_csv(r'./sim_result/base_29/S_EV100LIMIT30/result_1min_20.csv', sep=';', decimal=',', index_col=0)
    # df.index = pd.to_datetime(df.index)
    # df['distance'] = [0] + [round(e, 3) for e in np.diff(df['ref_distance'].to_numpy())]
    # df['errand'] = 0
    # df['hobby'] = 0
    # df['work'] = 0
    # distances = df['distance'].unique()
    #
    # for distance in distances:
    #     if distance > 0:
    #         # print(distance)
    #         tmp = df.loc[df['distance'] == distance].index
    #         for k in range(1, len(tmp)):
    #             minutes = (tmp[k] - tmp[k - 1]).seconds/60
    #             print(minutes)
    #             if minutes == 91:
    #                 df.loc[df['distance'] == distance, 'hobby'] = 1
    #                 x = df.loc[df['hobby'] == 1]
    #                 dates = np.unique(x.index.date)
    #                 for date in dates:
    #                     tmp = x.loc[x.index.date == date]
    #                     df.loc[tmp.index.min():tmp.index.max(), 'hobby'] = 100
    #                 break
    #             if minutes == 36:
    #                 df.loc[df['distance'] == distance, 'errand'] = 1
    #                 x = df.loc[df['errand'] == 1]
    #                 dates = np.unique(x.index.date)
    #                 for date in dates:
    #                     tmp = x.loc[x.index.date == date]
    #                     df.loc[tmp.index.min():tmp.index.max(), 'errand'] = 100
    #                 break
    #             elif minutes == 526 or minutes == 241:
    #                 df.loc[df['distance'] == distance, 'work'] = 1
    #                 x = df.loc[df['work'] == 1]
    #                 dates = np.unique(x.index.date)
    #                 for date in dates:
    #                     tmp = x.loc[x.index.date == date]
    #                     df.loc[tmp.index.min():tmp.index.max(), 'work'] = 100
    #                 break
    # df = df.loc[:, ['ref_soc', 'ref_distance', 'distance', 'errand', 'hobby', 'work']]
    # df = df.loc[(df.index >= pd.to_datetime('2022-01-03')) & (df.index < pd.to_datetime('2022-01-10'))]
    # df.to_csv('ref_car.csv', sep=';', decimal=',')
    #
    # pass



