from interfaces.results import Results
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

r = Results()
s_date = pd.to_datetime('2022-01-03')
e_date = pd.to_datetime('2022-01-10')

name = 'EV100'

font = dict(family="Verdana", size=10, color="black")

lines = {'Percentile 25 %': dict(color='rgba(0,128,0,1)', width=1.5, shape='hv'),
         'Percentile 75 %': dict(color='rgba(235,235,0,1)', width=1.5, shape='hv'),
         'Percentile 95 %': dict(color='rgba(255,153,0,1)', width=1.5, shape='hv'),
         'Average': dict(color='rgba(0,0,0,1)', width=1.5, shape='hv'),
         'Maximum': dict(color='rgba(255,0,0,1)', width=1.5, shape='hv')}

titles = {f'{name}CONTROL-FALSE': f'{name[:2]} {name[2:]} % - Without Grid Fee',
          f'{name}CONTROL-TRUE': f'{name[:2]} {name[2:]} % - Dynamic Grid Fee'}

sub_titles = 2 * [f"{titles[s]}" for s in [s for s in titles.keys()]]
fig = make_subplots(rows=2, cols=2, shared_xaxes=True, shared_yaxes=False, subplot_titles=sub_titles,
                    horizontal_spacing=0.1)
for col, scenario in zip([1, 2], [s for s in titles.keys()]):
    print(f'-> collecting data for {scenario}')
    for row, asset in zip([1, 2], ['line', 'transformer']):
        data = r.get_asset_type_util(asset=asset, scenario=scenario)
        for column in data.columns:
            fig.add_trace(go.Scatter(x=data.index,
                                     y=data[column],
                                     line=lines[column],
                                     showlegend=True if row == 1 and col == 1 else False,
                                     name=f"{column}"),
                          secondary_y=False, row=row, col=col)

            fig.add_trace(go.Scatter(x=data.index,
                                     y=[100 for _ in data.index],
                                     line=dict(color='rgba(166,166,166,1)', width=1, shape='hv'),
                                     showlegend=False, ),
                          secondary_y=False, row=row, col=col)

            fig.update_yaxes(title_text=f"Utilization {asset.capitalize()} [%]",
                             secondary_y=False,
                             showgrid=True,
                             gridwidth=0.1,
                             gridcolor='rgba(0, 0, 0, 0.5)',
                             row=row, col=col,
                             range=[0, 150],
                             dtick=25)

fig.update_xaxes(showgrid=True, gridwidth=0.1, gridcolor='rgba(0, 0, 0, 0.5)')
fig.update_layout(font=font, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="right", x=1))
fig.write_image(f'./images/{name}_utilization.svg', width=1200, height=600)
