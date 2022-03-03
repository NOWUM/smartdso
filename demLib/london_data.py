import pandas as pd
from sqlalchemy import create_engine
from tqdm import tqdm

pd.options.mode.chained_assignment = None  # default='warn'

engine = create_engine('postgresql://opendata:opendata@10.13.10.41:5432/londondatastore')

if __name__ == "__main__":
    df = pd.read_csv(r"C:\Users\rieke\Downloads\CC_LCL-FullData.csv")
    ids = df['LCLid'].unique()
    for id_ in tqdm(ids):
        data = df.loc[df['LCLid'] == id_]
        data.loc[:, 'DateTime'] = pd.to_datetime(data['DateTime'])
        data = data.set_index(['LCLid', 'DateTime'])
        data.columns = ['tariff', 'power']
        data['power'] = [float(value) / 0.5 for value in data['power'].str.strip().replace('Null', 0).values]
        data.to_sql('consumption', engine, if_exists='append')
