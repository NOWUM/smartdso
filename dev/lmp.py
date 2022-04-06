import plotly.graph_objects as go
import plotly.express as px
from plotly.offline import plot
import pandas as pd


if __name__ == "__main__":

    nodes = pd.read_csv(r'../gridLib/data/export/nodes.csv', index_col=0)
    lmp = pd.read_csv(r'./sim_result/S_EV100LIMIT30/lmp_1min_0.csv', sep=';', decimal=',', index_col=0)
    lmp.index = pd.to_datetime(lmp.index)
    lmp15 = lmp.resample('15min').mean()
    result = dict(time=[], lon=[], lat=[], price=[])

    x = lmp15.iloc[:5*96]

    for node in x.columns:
        coords = nodes.loc[node, ['lon', 'lat']]
        for r in x[node].iteritems():
            result['time'].append(str(r[0]))
            result['lon'].append(coords.lon)
            result['lat'].append(coords.lat)
            result['price'].append(r[1] + 24)

    df = pd.DataFrame(result)

    fig = px.scatter_mapbox(df, 'lat', 'lon', color='price', size='price',
                            height=1000, zoom=12,
                            mapbox_style='carto-positron', center={"lat": 50.805063, "lon": 6.49778},
                            title='LMPs', size_max=20,
                            range_color=[20.0, int(df['price'].max()) + 1],
                            animation_frame='time', animation_group='lat')

    plot(fig, 'test.html')