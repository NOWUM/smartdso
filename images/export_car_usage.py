from interfaces.results import Results
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import logging

logger = logging.getLogger('ImageExporter')

FONT = dict(family="Verdana", size=10, color="black")
ITERATIONS = [5,26]

RESULTS = Results()

FIG = make_subplots(rows=2, cols=1, specs=[[{"secondary_y": True}], [{"secondary_y": True}]],
                    shared_xaxes=True)
if __name__ == "__main__":
    for iteration, row in zip(ITERATIONS, [1,2]):
        df_car = RESULTS.get_cars(scenario='EV100CONTROL-TRUE', iteration=iteration)
        scale = df_car['odometer'].values.max()
        x_ticks = [i for i in range(len(df_car.index))]
        ticks = [t.strftime('%m.%d %H:%M') for t in df_car.index[::360]]
        # -> soc plot on first y axis
        FIG.add_trace(go.Scatter(x=x_ticks,
                                 y=df_car['soc'].values,
                                 name='SoC',
                                 line=dict(color='black', width=1.5, dash='dot'),
                                 showlegend=True if row == 1 else False),
                      secondary_y=False, col=1, row=row)
        # -> distance on second y axis
        FIG.add_trace(go.Scatter(x=x_ticks,
                                 y=df_car['odometer'].values,
                                 name='Distance',
                                 line=dict(color='red', width=1.5),
                                 showlegend=True if row == 1 else False),
                      secondary_y=True, col=1, row=row)
        # -> usage type on second y axis scaled with max distance
        FIG.add_trace(go.Scatter(x=x_ticks,
                                 y=df_car['work'].values * scale,
                                 name='Work',
                                 fill='tozeroy',
                                 line=dict(width=1, color='rgba(200, 0, 0, 0.1)'),
                                 showlegend=True if row == 1 else False),
                      secondary_y=True, col=1, row=row)
        FIG.add_trace(go.Scatter(x=x_ticks,
                                 y=df_car['errand'].values * scale,
                                 name='Errand',
                                 fill='tozeroy',
                                 line=dict(width=1, color='rgba(0, 150, 0, 0.1)'),
                                 showlegend=True if row == 1 else False),
                      secondary_y=True, col=1, row=row)
        FIG.add_trace(go.Scatter(x=x_ticks,
                                 y=df_car['hobby'].values * scale,
                                 name='Hobby',
                                 fill='tozeroy',
                                 line=dict(width=1, color='rgba(0, 0, 200, 0.1)'),
                                 showlegend=True if row == 1 else False),
                      secondary_y=True, col=1, row=row)
        # -> set axes titles
        FIG.update_yaxes(title_text="SoC [%]",
                         secondary_y=False,
                         showgrid=True,
                         gridwidth=0.1,
                         gridcolor='rgba(0, 0, 0, 0.5)',
                         dtick=25,
                         range=[0,101])
        FIG.update_yaxes(title_text="Distance [km]",
                         secondary_y=True)
        FIG.update_xaxes(showgrid=True,
                         gridwidth=0.1,
                         gridcolor='rgba(0, 0, 0, 0.5)',
                         tickmode='array',
                         tickvals=[t for t in range(0,len(df_car.index), 360)],
                         ticktext=ticks)


    FIG.update_layout(font=FONT, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    FIG.update_layout(legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="right", x=1))
    FIG.write_image(f'./images/car_usage.svg', width=1200, height=600)