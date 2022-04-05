import pandas as pd
from glob import glob as gb
import numpy as np

path = r'./sim_result/S_EV100LIMIT30'

if __name__ == "__main__":
    data = [pd.read_csv(file, sep=';', decimal=',', index_col=0) for file in gb(fr'{path}/result_1min_*.csv')]
    data_frames = dict()
    for column in data[0].columns:
        d_all = np.asarray([d[column].values for d in data]).T
        df = pd.DataFrame(dict(min=np.min(d_all, axis=1), avg=np.mean(d_all, axis=1),
                               max=np.max(d_all, axis=1)), index=data[0].index)
        #with pd.ExcelWriter(fr'{path}.xlsx', if_sheet_exists='replace', mode='a') as writer:
        #    df.to_excel(writer, sheet_name=column)
