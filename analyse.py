import pandas as pd
import sqlite3
import numpy
import seaborn as sns
import matplotlib.pyplot as plt

database = sqlite3.connect('result.db')


if __name__ == "__main__":
    # ---> get result data set
    query = 'SELECT * FROM results'
    df = pd.read_sql(query, database)
    df['index'] = pd.to_datetime(df['index'])
    df = df.set_index('index')

    # ---> get charged power for all EVs
    power = df['charged']
    power.index = df.index
    # ---> resample in 5 min and 15 min intervals
    power_5min = power.resample('5min').mean()
    power_15min = power.resample('15min').mean()
    # ---> save data
    power.to_csv('./results/charged_power_1min.csv', sep=';', decimal=',')
    power_5min.to_csv('./results/charged_power_5min.csv', sep=';', decimal=',')
    power_15min.to_csv('./results/charged_power_15min.csv', sep=';', decimal=',')

    # ---> get requests
    requests = df['requests']
    requests.index = df.index
    # ---> resample in 60 min intervals
    requests_ = requests.resample('60min').sum()
    requests_ = requests_.resample('5min').ffill()

    requests.to_csv('./results/requests_1min.csv', sep=';', decimal=',')
    requests_.to_csv('./results/requests_60min.csv', sep=';', decimal=',')

    # ---> get commits
    commits = df['commits']
    commits.index = df.index
    # ---> resample in 60 min intervals
    commits_ = commits.resample('60min').sum()
    commits_ = commits_.resample('5min').ffill()

    commits.to_csv('./results/commits_1min.csv', sep=';', decimal=',')
    commits_.to_csv('./results/commits_60min.csv', sep=';', decimal=',')

    # ---> get soc
    soc = df['soc']
    soc.index = df.index
    soc.to_csv('./results/soc.csv', sep=';', decimal=',')

    #sns.lineplot(data=requests_)
    #plt.show()