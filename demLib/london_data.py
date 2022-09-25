import pandas as pd
from sqlalchemy import create_engine
from tqdm import tqdm
import connectorx as cx
pd.options.mode.chained_assignment = None

# change to your running timescaledb
DATABASE_URI = 'postgresql://opendata:opendata@10.13.10.41:5432/londondatastore'
engine = create_engine(DATABASE_URI)

# download data from:
# https://data.london.gov.uk/download/smartmeter-energy-use-data-in-london-households
# extract and add to path variable

PATH = r"C:\Users\rieke\Downloads\CC_LCL-FullData.csv"


def read_csv_data(path: str):
    df = pd.read_csv(path)
    df['DateTime'] = pd.to_datetime(df['DateTime'], infer_datetime_format=True)
    data = df.set_index(['LCLid', 'DateTime'])

    data.columns = ['tariff', 'power']
    data['power'] = [float(value) / 0.5 for value in data['power'].str.strip().replace('Null', 0).values]

    return data


if __name__ == "__main__":

    df = pd.read_csv(r'./gridLib/data/grid_allocations.csv', index_col=0)
    ids = tuple(df['london_data'].unique())

    demand_data = {}
    query = f'SELECT "LCLid", "DateTime" as time, power from consumption where "LCLid" in {ids}  and' \
            f'"DateTime" >= \'2013-01-01 00:00\' and "DateTime" < \'2014-01-01 00:00\''
    print(query)
    data = cx.read_sql(conn=DATABASE_URI, query=query, return_type="pandas")


    # try:
    #     data_frame = read_csv_data(PATH)
    #     data_frame.to_sql('consumption', engine, if_exists='replace', chunksize=10000)
    # except Exception as e:
    #     print(repr(e))

