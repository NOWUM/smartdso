import pandas as pd
import plotly.offline
from plotly.subplots import make_subplots
import plotly.graph_objects as go

GRID_COLOR = 'rgba(0, 0, 0, 0.5)'
LINE_LAYOUT = dict(color='rgb(204, 0, 0)')
FONT = dict(family="Verdana", size=12, color="black")
FIGURE_LAYOUT = dict(font=FONT, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                     boxmode='group', boxgroupgap=0.005, boxgap=0.005, barmode='group',
                     bargap=0.005)

COLORS = {'pv_generation': 'rgb(255,128,0)', 'availability': 'rgb(0,0,0)', 'market_price': 'rgb(0,128,255)'}


def overview(data: pd.DataFrame, num_rows: int = 1) -> plotly.graph_objects.Figure:

    # -> crate figure with two sub plots
    fig = make_subplots(rows=num_rows, cols=1, specs=[num_rows*[{"secondary_y": True}]],
                        shared_xaxes=True)

    for column in data.columns:
        plt = go.Scatter(
            x=data.index,
            y=data.loc[:, column].values,
            line={'color': COLORS[column], 'width': 1.5},
            showlegend=True,
            name=f"{column}"
        )
        if column == 'market_price':
            fig.add_trace(plt, row=1, col=1, secondary_y=False)
        else:
            fig.add_trace(plt, row=1, col=1, secondary_y=True)

    # -> set axes titles
    fig.update_yaxes(title_text="Market Price [ct/KWh]",
                     secondary_y=False,
                     showgrid=True,
                     gridwidth=0.1,
                     gridcolor='rgba(0, 0, 0, 0.5)',
                     dtick=5,
                     range=[0, 30],
                     row=1, col=1)
    fig.update_yaxes(title_text="Availability / PV Generation [%]",
                     secondary_y=True, dtick=25, range=[0, 101],
                     row=1, col=1)
    fig.update_xaxes(showgrid=True,
                     gridwidth=0.1,
                     gridcolor='rgba(0, 0, 0, 0.5)')

    fig.update_layout(font=FONT, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="right", x=1))

    return fig
