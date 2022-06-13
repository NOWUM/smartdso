import pandas as pd
from sqlalchemy import create_engine
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

    try:
        data_frame = read_csv_data(PATH)
        data_frame.to_sql('consumption', engine, if_exists='replace', chunksize=10000)
    except Exception as e:
        print(repr(e))

