import pandas as pd
from sqlalchemy import create_engine
pd.options.mode.chained_assignment = None

engine = create_engine('postgresql://opendata:opendata@10.13.10.41:5432/londondatastore')


if __name__ == "__main__":
    df = pd.read_csv(r"C:\Users\rieke\Downloads\CC_LCL-FullData.csv")
    df['DateTime'] = pd.to_datetime(df['DateTime'])
    data = df.set_index(['LCLid', 'DateTime'])
    data.columns = ['tariff', 'power']
    data['power'] = [float(value) / 0.5 for value in data['power'].str.strip().replace('Null', 0).values]
    data.to_sql('consumption', engine, if_exists='replace', chunksize=10000)
