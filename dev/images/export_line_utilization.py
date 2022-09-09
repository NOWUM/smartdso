from dev.interfaces.results import Results
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import logging

logger = logging.getLogger('ImageExporter')

EV_RATIO = 'EV100'
FONT = dict(family="Verdana", size=10, color="black")
START_DATE, END_DATE = pd.to_datetime('2022-01-03'), pd.to_datetime('2022-01-10')

LINES = {'Percentile 25 %': dict(color='rgba(0,128,0,1)', width=1.5, shape='hv'),
         'Percentile 75 %': dict(color='rgba(235,235,0,1)', width=1.5, shape='hv'),
         'Percentile 95 %': dict(color='rgba(255,153,0,1)', width=1.5, shape='hv'),
         'Average': dict(color='rgba(0,0,0,1)', width=1.5, shape='hv'),
         'Maximum': dict(color='rgba(255,0,0,1)', width=1.5, shape='hv')}

TITLES = {f'{EV_RATIO}CONTROL-FALSE': f'{EV_RATIO[:2]} {EV_RATIO[2:]} % - Without Grid Fee',
          f'{EV_RATIO}CONTROL-TRUE': f'{EV_RATIO[:2]} {EV_RATIO[2:]} % - Dynamic Grid Fee'}

SUB_TITLES = 2 * [f"{TITLES[s]}" for s in [s for s in TITLES.keys()]]

RESULTS = Results()

FIG = make_subplots(rows=2, cols=2, shared_xaxes=True, shared_yaxes=False, subplot_titles=SUB_TITLES,
                    horizontal_spacing=0.1)

if __name__ == "__main__":

    for col, scenario in zip([1, 2], [s for s in TITLES.keys()]):
        logger.info(f'-> collecting data for {scenario}')
        for row, asset in zip([1, 2], ['inlet', 'outlet']):
            data = RESULTS.get_asset_type_util(asset=asset, scenario=scenario)
            for column in data.columns:
                FIG.add_trace(go.Scatter(x=data.index,
                                         y=data[column],
                                         line=LINES[column],
                                         showlegend=True if row == 1 and col == 1 else False,
                                         name=f"{column}"),
                              secondary_y=False, row=row, col=col)

                FIG.add_trace(go.Scatter(x=data.index,
                                         y=[100 for _ in data.index],
                                         line=dict(color='rgba(166,166,166,1)', width=1, shape='hv'),
                                         showlegend=False, ),
                              secondary_y=False, row=row, col=col)

                FIG.update_yaxes(title_text=f"Utilization {asset.capitalize()} [%]",
                                 secondary_y=False,
                                 showgrid=True,
                                 gridwidth=0.1,
                                 gridcolor='rgba(0, 0, 0, 0.5)',
                                 row=row, col=col,
                                 range=[0, 150],
                                 dtick=25)

    FIG.update_xaxes(showgrid=True, gridwidth=0.1, gridcolor='rgba(0, 0, 0, 0.5)')
    FIG.update_layout(font=FONT, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    FIG.update_layout(legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="right", x=1))
    FIG.write_image(f'./images/{EV_RATIO}_line_utilization.svg', width=1200, height=600)
