import numpy as np

from interfaces.results import Results
from plotLib.plot import plot_charge
import pandas as pd
from datetime import timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots

s_date = pd.to_datetime('2022-01-03')
e_date = pd.to_datetime('2022-01-10')

r = Results()
scenarios = [*r.scenarios]


font = dict(family="Verdana", size=10, color="black")


# -> add confidence_plot
def add_confidence(f: go.Figure, df: pd.DataFrame, name: str, row: int = 1, col: int = 1, secondary_y: bool = False):
    # -> minimal
    f.add_trace(go.Scatter(x=df.index,
                           y=df['min'].values,
                           line=dict(color='rgba(0,0,0,0.5)', width=0.2),
                           showlegend=False,
                           name=f"Minimal {name}"),
                secondary_y=secondary_y, row=row, col=col)


def plot_utilization(sc_data: list, titles: list):
    fig = make_subplots(rows=len(sc_data),
                        cols=2,
                        specs=[[{} for _ in range(len(sc_data))], [{"colspan": 1} for _ in range(len(sc_data))]],
                        subplot_titles=titles,
                        shared_xaxes=True)

    lines = {'Percentile 25 %': dict(color='rgba(0,128,0,1)', width=1.5, shape='hv'),
             'Percentile 75 %': dict(color='rgba(235,235,0,1)', width=1.5, shape='hv'),
             'Percentile 95 %': dict(color='rgba(255,153,0,1)', width=1.5, shape='hv'),
             'Average': dict(color='rgba(0,0,0,1)', width=1.5, shape='hv'),
             'Maximum': dict(color='rgba(255,0,0,1)', width=1.5, shape='hv')}

    for row, data in zip(range(1, len(sc_data) + 1), sc_data):
        for col, asset in zip([1, 2], ['line', 'transformer']):
            for column in data[asset].columns:
                fig.add_trace(go.Scatter(x=data[asset].index,
                                         y=data[asset][column],
                                         line=lines[column],
                                         showlegend=True if row == 1 and col == 1 else False,
                                         name=f"{column}"),
                              secondary_y=False, row=col, col=row)

                fig.update_yaxes(title_text=f"Utilization {asset.capitalize()} [%]",
                                 secondary_y=False,
                                 showgrid=True,
                                 gridwidth=0.1,
                                 gridcolor='rgba(0, 0, 0, 0.5)',
                                 row=col, col=row,
                                 range=[0, 150],
                                 dtick=25)

    fig.update_xaxes(showgrid=True, gridwidth=0.1, gridcolor='rgba(0, 0, 0, 0.5)')
    fig.update_layout(font=font, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')

    return fig


def plot_sorted_utilization(data: list, legend: list):
    fig = make_subplots(rows=1,
                        cols=1,
                        shared_xaxes=True)
    for d, l in zip(data, legend):
        fig.add_trace(go.Scatter(y=d, x=np.linspace(0, 100, len(d)), name=l), secondary_y=False, row=1, col=1)

    return fig

if __name__ == "__main__":
    # data, titles = [], []
    # for scenario in scenarios:
    #     if 'EV100' in scenario:
    #         print(f'-> collecting data for {scenario}')
    #         r.scenario = scenario
    #         data += [{asset: r.get_asset_type_util(asset=asset) for asset in ['line', 'transformer']}]
    #         if "TRUE" in scenario:
    #             titles += [f'EV 100 % + Dynamic Grid Fee']
    #         else:
    #             titles += [f'EV 100 % + Without Grid Fee']
    # fig = plot_utilization(data, titles)
    # fig.write_image('test.svg', width=1200, height=600)

    data, titles = [], []
    for scenario in scenarios:
        r.scenario = scenario
        if 'EV100' in scenario:
            data += [r.get_sorted_utilization()[::10000]]
            if "TRUE" in scenario:
                titles += [f'EV 100 % - Dynamic Grid Fee']
            else:
                titles += [f'EV 100 % - Without Grid Fee']

    fig = plot_sorted_utilization(data, titles)
    fig.write_image('test1.svg')
