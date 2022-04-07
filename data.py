import pandas as pd
from collections import defaultdict
import os.path
import numpy as np
import paramiko

pk = paramiko.Ed25519Key.from_private_key(open(r'C:\Users\rieke\.ssh\id_ed25519'))
path = 'smartdso/sim_result/'
scenarios = dict(S_EV100LIMIT30="10.13.10.54", S_EV80LIMIT30="10.13.10.55", S_EV50LIMIT30="10.13.10.56")


def meta_analyze():
    results = defaultdict(list)

    for key, value in scenarios.items():
        s_path = f'{path}{key}'
        host, port = value, 22
        transport = paramiko.Transport((host, port))
        transport.connect(username='nowum', pkey=pk)
        sftp = paramiko.SFTPClient.from_transport(transport)
        sftp.chdir(s_path)
        data_frames = []
        for file in sftp.listdir():
            if 'result_1min' in file:
                with sftp.open(file) as f:
                    data_frames.append(pd.read_csv(f, sep=';', decimal=',', index_col=0))
        for column in data_frames[0].columns:
            d_all = np.asarray([d[column].values for d in data_frames]).T
            avg, min_, max_ = np.mean(d_all, axis=1), np.min(d_all, axis=1), np.max(d_all, axis=1)
            columns = pd.MultiIndex.from_product([[key.split('_')[-1].split('LIMIT')[0]], ['avg', 'low', 'up']])
            df = pd.DataFrame(dict(avg=avg, low=min_, up=max_-min_), index=data_frames[0].index)
            df.columns = columns
            results[column].append(df)

        sftp.close()
        transport.close()

    parameters = ['charged', 'price', 'shift', 'soc', 'empty', 'utilization', 'concurrency', 'waiting']
    if not os.path.exists('meta_results.xlsx'):
        pd.DataFrame().to_excel('meta_results.xlsx')

    with pd.ExcelWriter(fr'meta_results.xlsx', if_sheet_exists='replace', mode='a') as writer:
        for key, dataframes in results.items():
            if key in parameters:
                s_time, e_time = pd.to_datetime('2022-01-03'), pd.to_datetime('2022-01-10 23:59:59')
                df = pd.concat(dataframes, axis=1)
                df.index = [pd.to_datetime(i) for i in df.index]
                df = df.resample('5min').mean()
                df.loc[s_time:e_time].to_excel(writer, sheet_name=key)


def car_analyze():

    key = 'S_EV100LIMIT30'
    s_path = f'{path}{key}'
    host, port = scenarios[key], 22
    transport = paramiko.Transport((host, port))
    transport.connect(username='nowum', pkey=pk)
    sftp = paramiko.SFTPClient.from_transport(transport)
    sftp.chdir(s_path)
    data_frames = []
    for file in sftp.listdir():
        if 'result_1min' in file:
            with sftp.open(file) as f:
                data_frames.append(pd.read_csv(f, sep=';', decimal=',', index_col=0))
    sftp.close()
    transport.close()

    if not os.path.exists('car_results.xlsx'):
        pd.DataFrame().to_excel('car_results.xlsx')

    index = 0
    mean_distance = []
    with pd.ExcelWriter(fr'car_results.xlsx', if_sheet_exists='replace', mode='a') as writer:
        for df in data_frames:
            df.index = pd.to_datetime(df.index)
            df['distance'] = [0] + [round(e, 3) for e in np.diff(df['ref_distance'].to_numpy())]
            df['errand'], df['hobby'], df['work'] = 0, 0, 0
            distances = df['distance'].unique()

            for distance in distances:
                if distance > 0:
                    tmp = df.loc[df['distance'] == distance].index
                    for k in range(1, len(tmp)):
                        minutes = (tmp[k] - tmp[k - 1]).seconds / 60
                        if minutes == 91:
                            df.loc[df['distance'] == distance, 'hobby'] = 1
                            x = df.loc[df['hobby'] == 1]
                            dates = np.unique(x.index.date)
                            for date in dates:
                                tmp = x.loc[x.index.date == date]
                                df.loc[tmp.index.min():tmp.index.max(), 'hobby'] = 100
                            break
                        if minutes == 36:
                            df.loc[df['distance'] == distance, 'errand'] = 1
                            x = df.loc[df['errand'] == 1]
                            dates = np.unique(x.index.date)
                            for date in dates:
                                tmp = x.loc[x.index.date == date]
                                df.loc[tmp.index.min():tmp.index.max(), 'errand'] = 100
                            break
                        elif minutes == 526 or minutes == 241:
                            df.loc[df['distance'] == distance, 'work'] = 1
                            x = df.loc[df['work'] == 1]
                            dates = np.unique(x.index.date)
                            for date in dates:
                                tmp = x.loc[x.index.date == date]
                                df.loc[tmp.index.min():tmp.index.max(), 'work'] = 100
                            break

            df = df.loc[:, ['ref_soc', 'ref_distance', 'errand', 'hobby', 'work']]
            df.columns = ['SoC', 'Distance', 'Errand', 'Hobby', 'Work']

            mean_distance.append(df['Distance'].values[-1] - df['Distance'].values[0])

            df = df.loc[(df.index >= pd.to_datetime('2022-01-03')) & (df.index < pd.to_datetime('2022-01-10'))]
            df.to_excel(writer, sheet_name=f'car_{index}')
            index += 1

    print(np.mean(mean_distance))


if __name__ == "__main__":
    meta_analyze()
    car_analyze()
