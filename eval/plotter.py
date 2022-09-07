import pandas as pd
from plotly.subplots import make_subplots
import plotly.graph_objects as go

GRID_COLOR = 'rgba(0, 0, 0, 0.5)'
LINE_LAYOUT = dict(color='rgb(204, 0, 0)')
FONT = dict(family="Verdana", size=12, color="black")
FIGURE_LAYOUT = dict(font=FONT, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                     boxmode='group', boxgroupgap=0.005, boxgap=0.005, barmode='group',
                     bargap=0.005)

COLORS_SENS = {1.0: 'rgb(0,204,0)', 1.7: 'rgb(0,204,153)', 2.7: 'rgb(0,102,153)', 4.0: 'rgb(0,0,153)'}


def plot_mean_charging(data: pd.DataFrame):
    fig = make_subplots(rows=1, cols=1, specs=[[{"secondary_y": False}]])

    for column in data.columns:
          fig.add_trace(go.Scatter(name=f'mean charging {column}',
                      x=data.index, y=data[column].values,
                      mode='lines'), row=1, col=1)

    fig.update_yaxes(title_text="Mean Charging Power [MW]",
                     showgrid=True, gridwidth=0.1, gridcolor=GRID_COLOR)

    fig.update_xaxes(title_text="Intervals [HH:MM]",
                     showgrid=True, gridwidth=0.1, gridcolor=GRID_COLOR)

    fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=-0.1, xanchor="right", x=1))
    fig.update_layout(**FIGURE_LAYOUT)

    return fig

def create_price_sensitivity_plot():
    price = get_prices()
    prc = price.sort_values('price', ascending=False).values.flatten() + 0.11
    sens = get_price_sensitivities()

    fig = make_subplots(rows=1, cols=1, specs=[[{"secondary_y": True}]], shared_xaxes=True)
    for s, name in zip(sens, [1.0, 1.7, 2.7, 4.0]):
        fig.add_trace(go.Scatter(
            x=np.linspace(0, 100, len(s)),
            y=100 * s,
            mode='lines',
            name=f'Sensitivity {name}',
            line=dict(color=COLORS_SENS[name])
        ), col=1, row=1, secondary_y=True)

    fig.add_trace(go.Scatter(
        x=np.linspace(0, 100, len(prc)),
        y=prc,
        mode='lines',
        name=f'DayAhead Price',
        line=dict(color='rgb(204,0,0)')
    ), col=1, row=1, secondary_y=False)

    # Here we modify the tickangle of the xaxis, resulting in rotated labels.
    fig.update_yaxes(title_text="Price [ct/kWh]",
                     showgrid=True,
                     gridwidth=0.1,
                     range=[0, 90],
                     gridcolor='rgba(0, 0, 0, 0.5)')
    fig.update_yaxes(title_text="Marginal utility [ct/dSoC]",
                     showgrid=False,
                     secondary_y=True,
                     gridwidth=0.1,
                     range=[0, 90],
                     gridcolor='rgba(0, 0, 0, 0.5)')
    fig.update_xaxes(title_text="SoC [%]",
                     showgrid=True,
                     gridwidth=0.1,
                     gridcolor='rgba(0, 0, 0, 0.5)')