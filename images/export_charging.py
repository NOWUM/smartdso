from interfaces.results import Results
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# -> add confidence_plot
def add_confidence(f: go.Figure, df: pd.DataFrame, name: str, row: int = 1, col: int = 1, secondary_y: bool = False):
    # -> minimal
    f.add_trace(go.Scatter(x=df.index,
                           y=df['min'].values,
                           line=dict(color='rgba(0,0,0,0.5)', width=0.2),
                           showlegend=False,
                           name=f"Minimal {name}"),
                secondary_y=secondary_y, row=row, col=col)

    # -> maximal
    f.add_trace(go.Scatter(x=df.index,
                           y=df['max'].values,
                           line=dict(color='rgba(0,0,0,0.5)', width=0.2),
                           fillcolor='rgba(50, 50, 50, 0.1)',
                           fill='tonexty',
                           showlegend=False,
                           name=f"Maximal {name}"),
                secondary_y=secondary_y, row=row, col=col)

    return f

FONT = dict(family="Verdana", size=10, color="black")

RESULTS = Results()

if __name__ == '__main__':

    # -> get simulation results
    sim_result = RESULTS.get_vars('EV100CONTROL-TRUE')

    charged = sim_result.loc[:, 'charged']
    shifted = sim_result.loc[:, ['avg_shifted', 'min_shifted', 'max_shifted']]
    shifted.columns = map(lambda x: x.split('_')[0], shifted.columns)

    # -> crate figure with two sub plots
    FIG = make_subplots(rows=1, cols=1, specs=[[{"secondary_y": True}]],
                        shared_xaxes=True)
    # -> price plot
    # -> charge and shift plot
    # -> charging
    FIG.add_trace(go.Scatter(x=charged.index, y=charged.values.flatten(),
                             line=dict(color='rgba(255,0,0,1)', width=1.5),
                             showlegend=True, name=f"Charged"),
                  secondary_y=False, row=1, col=1)
    # -> shift
    FIG.add_trace(go.Scatter(x=shifted.index, y=shifted['avg'].values,
                             line=dict(color='rgba(0,0,0,1)', width=1.5),
                             showlegend=True, name="Shifted"),
                  secondary_y=True, row=1, col=1)
    FIG = add_confidence(FIG, shifted.loc[:, ['min', 'max']], name='Shift', row=1, col=1, secondary_y=True)

    # -> set axes titles
    FIG.update_yaxes(title_text="Charged Power [kW]",
                     secondary_y=False,
                     showgrid=True,
                     gridwidth=0.1,
                     gridcolor='rgba(0, 0, 0, 0.5)',
                     dtick=50,
                     range=[0,401],
                     row=1, col=1)
    FIG.update_yaxes(title_text="Shifted Power [kW]",
                     secondary_y=True, dtick=25, range=[0,101],
                     row=1, col=1)
    FIG.update_xaxes(showgrid=True,
                     gridwidth=0.1,
                     gridcolor='rgba(0, 0, 0, 0.5)')

    FIG.update_layout(font=FONT, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    FIG.update_layout(legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="right", x=1))
    FIG.write_image(f'./images/charging.svg', width=1200, height=600)