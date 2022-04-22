from plotly.subplots import make_subplots
import plotly.graph_objects as go
import pydeck as pdk
import pandas as pd
from gridLib.model import GridModel

api_key = 'pk.eyJ1Ijoicmlla2VjaCIsImEiOiJjazRiYTdndXkwYnN3M2xteGN2MHhtZjB0In0.33tSDK45TXF3lb3-G147jw'
font = dict(family="Verdana", size=12, color="black")
grid = GridModel()


def plot_charging(df_charge: pd.DataFrame, df_shift: pd.DataFrame, df_price: pd.DataFrame):
    # -> crate figure with two sub plots
    fig = make_subplots(rows=2, cols=1, specs=[[{"secondary_y": False}], [{"secondary_y": True}]],
                        shared_xaxes=True)
    for column in df_price.columns:
        fig.add_trace(go.Scatter(x=df_price.index, y=df_price[column].values + 29,
                                 name=f"Price {column}"), secondary_y=False, row=1, col=1)

        fig.add_trace(go.Scatter(x=df_charge.index, y=df_charge[column].values,
                                 name=f"Power Charged {column}"), secondary_y=False, row=2, col=1)

        fig.add_trace(go.Scatter(x=df_shift.index, y=df_shift[column].values,
                                 name=f"Power Shift {column}"), secondary_y=True, row=2, col=1)

    fig.update_yaxes(title_text="SoC [%]", secondary_y=False)
    fig.update_yaxes(title_text="Distance [km]", secondary_y=True)
    fig.update_layout(font=font, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    return fig


def plot_car(df_car: pd.DataFrame):
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    scale = df_car['odometer'].values.max()
    fig.add_trace(go.Scatter(x=df_car.index, y=df_car['soc'].values, name='SoC',
                             line=dict(color='black', width=3, dash='dot')), secondary_y=False)
    fig.add_trace(go.Scatter(x=df_car.index, y=df_car['odometer'].values, name='Distance',
                             line=dict(color='red', width=3)), secondary_y=True)
    fig.add_trace(go.Scatter(x=df_car.index, y=df_car['work'].values * scale, name='Work',
                             fill='tozeroy', mode='lines',
                             line=dict(width=0.5, color='rgba(200, 0, 0, 0.1)')), secondary_y=True)
    fig.add_trace(go.Scatter(x=df_car.index, y=df_car['errand'].values * scale, name='Errand',
                             fill='tozeroy', mode='lines',
                             line=dict(width=0.5, color='rgba(0, 150, 0, 0.1)')), secondary_y=True)
    fig.add_trace(go.Scatter(x=df_car.index, y=df_car['hobby'].values * scale, name='Hobby',
                             fill='tozeroy', mode='lines',
                             line=dict(width=0.5, color='rgba(0, 0, 200, 0.1)')), secondary_y=True)

    fig.update_yaxes(title_text="SoC [%]", secondary_y=False)
    fig.update_yaxes(title_text="Distance [km]", secondary_y=True)
    fig.update_layout(font=font, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')

    return fig


def plot_grid(sub_id: str = 'total'):
    view_state = pdk.ViewState(latitude=50.8076277635652, longitude=6.492433919140283, zoom=13)
    edges = pdk.Layer(type="PathLayer", data=grid.get_components(type_='edges', grid=sub_id), pickable=True,
                      width_min_pixels=1, get_path="geometry.coordinates", get_color=[0, 0, 0])
    nodes = pdk.Layer(type="ScatterplotLayer", data=grid.get_components(type_='nodes', grid=sub_id), pickable=True,
                      radius_min_pixels=2, radius_max_pixels=5, get_position="geometry.coordinates",
                      get_color=[255, 0, 0])
    return pdk.Deck(layers=[edges, nodes], initial_view_state=view_state,
                    tooltip={"html": "<b>Name: </b> {name} <br /> "},
                    api_keys=dict(mapbox=api_key), map_provider='mapbox', map_style='road')
