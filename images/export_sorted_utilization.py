from interfaces.results import Results
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


r = Results()
s_date = pd.to_datetime('2022-01-03')
e_date = pd.to_datetime('2022-01-10')

font = dict(family="Verdana", size=10, color="black")

names = {f'EV100CONTROL-FALSE': f'EV 100 % - Without Grid Fee',
         f'EV100CONTROL-TRUE': f'EV 100 % - Dynamic Grid Fee',
         f'EV80CONTROL-FALSE': f'EV 80 % - Without Grid Fee'}
         #f'EV50CONTROL-FALSE': f'EV 50 % - Without Grid Fee'}

lines = {f'EV100CONTROL-FALSE':dict(color='rgba(135,0,0,1)', width=1.5),
         f'EV100CONTROL-TRUE':dict(color='rgba(0,135,0,1)', width=1.5),
         f'EV80CONTROL-FALSE':dict(color='rgba(0,0,135,1)', width=1.5)
         }

fig = make_subplots(rows=1, cols=1, shared_xaxes=True, shared_yaxes=False)

for key in names.keys():

    data = r.get_sorted_utilization(scenario=key)[::10000]
    fig.add_trace(go.Scatter(x=np.linspace(0,100,len(data)),
                             y=data,
                             line=lines[key],
                             showlegend=True,
                             name=names[key]),
                  secondary_y=False, row=1, col=1)

    fig.add_trace(go.Scatter(
        x=[0],
        y=[data[0]],
        mode="markers+text",
        text=[names[key]],
        line=lines[key],
        textposition="bottom right",
        showlegend=False),
        row=1,
        col=1,
        secondary_y=False
    )

fig.update_yaxes(title_text=f"Utilization [%]", showgrid=True, gridwidth=0.1, gridcolor='rgba(0, 0, 0, 0.5)',
                 range=[0, 150], dtick=25)
fig.update_xaxes(showgrid=True, gridwidth=0.1, gridcolor='rgba(0, 0, 0, 0.5)', title_text=f"Number [%]")
fig.update_layout(font=font, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="right",x=1))

fig.write_image(f'./images/sorted_utilization.svg', width=1200, height=600)


