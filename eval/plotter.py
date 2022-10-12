import pandas as pd
import plotly.offline
from plotly.subplots import make_subplots
import plotly.graph_objects as go

GRID_COLOR = 'rgba(0, 0, 0, 0.5)'
LINE_LAYOUT = dict(color='rgb(204, 0, 0)')
FONT = dict(family="Verdana", size=8, color="black")
FIGURE_LAYOUT = dict(font=FONT, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                     boxmode='group', boxgroupgap=0.005, boxgap=0.005, barmode='group',
                     bargap=0.005)

COLORS = {'pv_generation': 'rgba(230,230,74,0.65)',
          'availability': 'rgba(0,0,0,0.6)',
          'market_price': 'rgba(255,0,0,0.7)',
          'charging': 'rgba(255,0,0,0.7)',
          'usage': 'rgba(0,0,0,0.6)',
          'soc': 'rgba(20,35,90,1.0)',
          'used_pv_generation': 'rgba(230,230,74,0.65)'
          }


def get_overview(f: plotly.graph_objects.Figure, data: pd.DataFrame, row: int = 1) -> plotly.graph_objects.Figure:
    for column in data.columns:
        plt = go.Scatter(
            x=data.index,
            y=data.loc[:, column].values,
            line={'color': COLORS[column], 'width': 1.5},
            showlegend=True,
            name=f"{column}"
        )
        if column == 'market_price':
            f.add_trace(plt, row=row, col=1, secondary_y=False)
        else:
            f.add_trace(plt, row=row, col=1, secondary_y=True)

    return f


def get_ev(f: plotly.graph_objects.Figure, data: pd.DataFrame, row: int = 1) -> plotly.graph_objects.Figure:
    # order is important for stacked and filled plots
    for column in ['used_pv_generation', 'charging', 'soc', 'usage']:
        plt_data = dict(x=data.index,
                        y=data.loc[:, column].values,
                        line={'color': COLORS[column], 'width': 1, "shape": 'hv'},
                        showlegend=True if row <= 2 else False,
                        name=f"{column}")
        if column in ['charging', 'used_pv_generation']:
            plt_data['fill'] = 'tonexty'

        plt = go.Scatter(**plt_data)

        if column in ['charging', 'used_pv_generation']:
            f.add_trace(plt, row=row, col=1, secondary_y=False)
        else:
            f.add_trace(plt, row=row, col=1, secondary_y=True)

    return f


def scenario_compare(data: dict, num_rows: int = 4) -> plotly.graph_objects.Figure:
    # -> crate figure with two sub plots
    fig = make_subplots(rows=num_rows, cols=1, specs=num_rows * [[{"secondary_y": True}]],
                        shared_xaxes=True, subplot_titles=[*data.keys()])

    row_counter = 1
    for key, d in data.items():
        if key == 'overview':
            fig = get_overview(fig, d, row_counter)
        else:
            fig = get_ev(fig, d, row_counter)
        row_counter += 1

    # -> set axes titles

    # -> first y axes
    fig.update_yaxes(title_text="Price [ct/kWh]",
                     secondary_y=False,
                     showgrid=True,
                     gridwidth=0.1,
                     gridcolor=GRID_COLOR,
                     dtick=5,
                     range=[0, 30],
                     row=1, col=1)

    for row in range(2, num_rows + 1):
        row_options = dict(secondary_y=False,
                           showgrid=True,
                           gridwidth=0.1,
                           gridcolor=GRID_COLOR,
                           dtick=5,
                           range=[0, 20],
                           row=row, col=1)
        if row == num_rows - 1:
            row_options['title_text'] = "Charging [kW]"
        fig.update_yaxes(**row_options)

    # -> second y axis
    fig.update_yaxes(title_text="Availability [%]",
                     secondary_y=True,
                     dtick=25,
                     range=[0, 101],
                     row=1, col=1)
    for row in range(2, num_rows + 1):
        row_options = dict(secondary_y=True,
                           showgrid=True,
                           dtick=0.25,
                           range=[0, 1],
                           row=row, col=1)
        if row == num_rows - 1:
            row_options['title_text'] = "Usage / SoC [%]"
        fig.update_yaxes(**row_options)

    # -> single x axis
    fig.update_xaxes(showgrid=True,
                     gridwidth=0.1,
                     gridcolor='rgba(0, 0, 0, 0.5)')

    fig.update_annotations(font_size=8)
    fig.update_layout(font=FONT, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="right", x=1))

    return fig


def overview(data: pd.DataFrame, num_rows: int = 1) -> plotly.graph_objects.Figure:
    # -> crate figure with two sub plots
    fig = make_subplots(rows=num_rows, cols=1, specs=[num_rows * [{"secondary_y": True}]],
                        shared_xaxes=True)

    fug = get_overview(fig, data, 1)

    # -> set axes titles
    fig.update_yaxes(title_text="Market Price [ct/kWh]",
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


def ev_plot(data: pd.DataFrame, num_rows: int = 1) -> plotly.graph_objects.Figure:
    # -> crate figure with two sub plots
    fig = make_subplots(rows=num_rows, cols=1, specs=[num_rows * [{"secondary_y": True}]],
                        shared_xaxes=True)

    fig = get_ev(fig, data, 1)

    # -> set axes titles
    fig.update_yaxes(title_text="Charging [kW]",
                     secondary_y=False,
                     showgrid=True,
                     gridwidth=0.1,
                     gridcolor='rgba(0, 0, 0, 0.5)',
                     dtick=5,
                     range=[0, 11],
                     row=1, col=1)
    fig.update_yaxes(title_text="Usage / SoC [%]",
                     secondary_y=True, dtick=0.25, range=[0, 1.01],
                     row=1, col=1)
    fig.update_xaxes(showgrid=True,
                     gridwidth=0.1,
                     gridcolor='rgba(0, 0, 0, 0.5)')

    fig.update_layout(font=FONT, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="right", x=1))

    return fig

def get_table(df: pd.DataFrame):
    fig = go.Figure(data=[go.Table(
        header=dict(values=['Description']+list(df.columns),
                    line_color='darkslategray',
                    fill_color='skyblue',
                    align='left',
                    height=30),
        cells=dict(values=[df.index] +[df.get(c) for c in df.columns],
                line_color='darkslategray',
                fill=dict(color=['paleturquoise']+['white']*len(df.columns)),
                align='left',
                height=20),
                )
    ])
    fig.update_layout(font=FONT, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="right", x=1))
    fig.update_layout(width=800, height=900)
    return fig