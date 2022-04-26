from plotly.subplots import make_subplots
import plotly.graph_objects as go
import pydeck as pdk
import pandas as pd
import numpy as np
from gridLib.model import GridModel
from matplotlib import cm

# -> mapbox token
api_key = 'pk.eyJ1Ijoicmlla2VjaCIsImEiOiJjazRiYTdndXkwYnN3M2xteGN2MHhtZjB0In0.33tSDK45TXF3lb3-G147jw'
# -> init state for pydeck map
view_state = pdk.ViewState(latitude=50.8076277635652, longitude=6.492433919140283, zoom=13)
# -> plotly font style
font = dict(family="Verdana", size=12, color="black")
# -> grid reference
grid = GridModel()
# -> colormap for utilization
c_map = cm.get_cmap('RdYlGn')


# -> get rgb color
def get_color(util):
    # c_map return red for zero and green for one
    color = np.asarray(c_map((1 - util / 100)))
    return [int(255 * c) for c in color[0:3]]


def plot_charge(df_charge: pd.DataFrame, df_shift: pd.DataFrame, df_price: pd.DataFrame):
    # -> crate figure with two sub plots
    fig = make_subplots(rows=2, cols=1, specs=[[{"secondary_y": False}], [{"secondary_y": True}]],
                        shared_xaxes=True)
    # -> price plot
    fig.add_trace(go.Scatter(x=df_price.index, y=df_price.values.flatten() + 29,
                             name=f"Price"), secondary_y=False, row=1, col=1)
    # -> charge and shift plot
    fig.add_trace(go.Scatter(x=df_charge.index, y=df_charge.values.flatten(),
                             name=f"Charged"), secondary_y=False, row=2, col=1)
    fig.add_trace(go.Scatter(x=df_shift.index, y=df_shift.values.flatten(),
                             name=f"Shifted"), secondary_y=True, row=2, col=1)
    # -> set axes titles
    fig.update_yaxes(title_text="Price [ct/kWh]", secondary_y=False, row=1, col=1)
    fig.update_yaxes(title_text="Charged Power [kW]", secondary_y=False, row=2, col=1)
    fig.update_yaxes(title_text="Shifted Power [kW]", secondary_y=True, row=2, col=1)
    fig.update_layout(font=font, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')

    return fig


def plot_car_usage(df_car: pd.DataFrame):
    scale = df_car['odometer'].values.max()
    # -> crate figure with secondary y axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    # -> soc plot on first y axis
    fig.add_trace(go.Scatter(x=df_car.index, y=df_car['soc'].values, name='SoC',
                             line=dict(color='black', width=3, dash='dot')), secondary_y=False)
    # -> distance on second y axis
    fig.add_trace(go.Scatter(x=df_car.index, y=df_car['odometer'].values, name='Distance',
                             line=dict(color='red', width=3)), secondary_y=True)
    # -> usage type on second y axis scaled with max distance
    fig.add_trace(go.Scatter(x=df_car.index, y=df_car['work'].values * scale, name='Work',
                             fill='tozeroy', mode='lines',
                             line=dict(width=0.5, color='rgba(200, 0, 0, 0.1)')), secondary_y=True)
    fig.add_trace(go.Scatter(x=df_car.index, y=df_car['errand'].values * scale, name='Errand',
                             fill='tozeroy', mode='lines',
                             line=dict(width=0.5, color='rgba(0, 150, 0, 0.1)')), secondary_y=True)
    fig.add_trace(go.Scatter(x=df_car.index, y=df_car['hobby'].values * scale, name='Hobby',
                             fill='tozeroy', mode='lines',
                             line=dict(width=0.5, color='rgba(0, 0, 200, 0.1)')), secondary_y=True)
    # -> set axes titles
    fig.update_yaxes(title_text="SoC [%]", secondary_y=False)
    fig.update_yaxes(title_text="Distance [km]", secondary_y=True)
    fig.update_layout(font=font, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')

    return fig


def plot_grid(line_utilization: pd.DataFrame = None, sub_id: str = 'total'):
    layers = []

    edges = grid.get_components(type_='edges', grid=sub_id)
    nodes = grid.get_components(type_='nodes', grid=sub_id)
    nodes['utilization'] = 0
    transformers = grid.get_components(type_='transformers', grid=sub_id)
    transformers['utilization'] = 0

    if line_utilization is None:
        layers += [pdk.Layer(type="PathLayer", data=edges, pickable=True, width_min_pixels=1,
                             get_path="geometry.coordinates", get_color=[0, 0, 0])]
    else:
        # -> get color for each line
        edges = grid.get_components(type_='edges', grid=sub_id)
        edges = edges.set_index('name').join(line_utilization, on='name')
        edges['color'] = edges['utilization'].apply(get_color)
        edges['utilization'] = edges['utilization'].apply(lambda x: round(x, 2))
        edges = edges.reset_index()

        layers += [pdk.Layer(type="PathLayer", data=edges, pickable=True, width_min_pixels=1,
                             get_path="geometry.coordinates", get_color='color')]

    layers += [pdk.Layer(type="ScatterplotLayer", data=nodes, pickable=True,
                         radius_min_pixels=2, radius_max_pixels=5, get_position="geometry.coordinates",
                         get_color=[0, 0, 255])]

    layers += [pdk.Layer(type="ScatterplotLayer", data=transformers, pickable=True,
                         radius_min_pixels=2, radius_max_pixels=5, get_position="geometry.coordinates",
                         get_color=[0, 0, 127])]

    return pdk.Deck(layers=layers, initial_view_state=view_state, api_keys=dict(mapbox=api_key), map_provider='mapbox',
                    map_style='road', tooltip={"html": "<b>Name: </b> {name} <br /> "
                                                       "<b>Utilization: </b> {utilization} %<br />"},)
