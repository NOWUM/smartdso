import pandas as pd


import numpy as np
import paramiko

pk = paramiko.Ed25519Key.from_private_key(open(r'C:\Users\rieke\.ssh\id_ed25519'))
path = 'smartdso/sim_result/'

scenarios = dict(
    S_EV100LIMIT30="10.13.10.54",
    S_EV80LIMIT30="10.13.10.55",
    S_EV50LIMIT30="10.13.10.56",
)

results = dict(
    S_EV100LIMIT30=[],
    S_EV80LIMIT30=[],
    S_EV50LIMIT30=[],
)

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
        df = pd.DataFrame(dict(min=np.min(d_all, axis=1), avg=np.mean(d_all, axis=1),
                               max=np.max(d_all, axis=1)), index=data_frames[0].index)
        results[key].append(df)

    sftp.close()
    transport.close()

