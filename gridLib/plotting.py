import plotly.graph_objs as go
import numpy as np
import pandas as pd
from shapely.wkt import loads

COLOR = {'0.4': '#000000', '10': '#278511', '20': '#222233', '110': '#424949'}
SIZES = {'0.4': 6, '10': 7, '20': 9, '110': 10}
API_KEY = 'pk.eyJ1Ijoicmlla2VjaCIsImEiOiJjazRiYTdndXkwYnN3M2xteGN2MHhtZjB0In0.33tSDK45TXF3lb3-G147jw'


def get_plot(nodes: pd.DataFrame = pd.DataFrame(),
             edges: pd.DataFrame = pd.DataFrame(),
             transformers: pd.DataFrame = pd.DataFrame(),
             consumers: pd.DataFrame = pd.DataFrame(),
             voltage_levels: tuple = (0.4,)) -> go.Figure:

    def get_edges(v_level: float = 0.4) -> go.Scattermapbox:
        lon_coords, lat_coords, names = [], [], []
        lines = edges[edges['v_nom'] == v_level]

        for name, attributes in lines.iterrows():
            shape = attributes['shape']
            x, y = shape.xy
            lat_coords, lon_coords = np.append(lat_coords, y), np.append(lon_coords, x)
            names = np.append(names, [f'<b>Name:</b> {name} <br />'])
            lat_coords, lon_coords = np.append(lat_coords, [None]), np.append(lon_coords, [None])
            names = np.append(names, [None])

        grid_plot = go.Scattermapbox(name=f'lines', hoverinfo='text', hovertext=names, mode='lines',
                                     lon=lon_coords, lat=lat_coords, line=dict(width=2, color=COLOR[str(v_level)]))
        return grid_plot

    def get_nodes(v_level: float = 0.4) -> go.Scattermapbox:
        busses = nodes.loc[nodes['v_nom'] == v_level]
        grid_plot = go.Scattermapbox(name=f'Nodes', mode='markers',
                                     lon=busses['lon'], lat=busses['lat'],
                                     hoverinfo='text', hovertext=[f'<b>Name:</b> {index}' for index in busses.index],
                                     marker=dict(size=SIZES[str(v_level)], color=COLOR[str(v_level)]))
        return grid_plot

    def get_consumers(v_level: float = 0.4) -> go.Scattermapbox:
        demand = consumers.loc[consumers['v_nom'] == v_level]
        grid_plot = go.Scattermapbox(name=f'Consumers on {v_level} kV', mode='markers',
                                     lon=demand['lon'], lat=demand['lat'],
                                     hoverinfo='text', hovertext=[f'<b>Name:</b> {index}' for index in demand.index])
        return grid_plot

    def get_transformers(v_level: float = 0.4) -> go.Scattermapbox:
        trans = transformers.loc[transformers['v0'] == v_level]
        grid_plot = go.Scattermapbox(name=f'Transformers', mode='markers',
                                     lon=trans['lon'], lat=trans['lat'], hoverinfo='text',
                                     hovertext=[f'<b>Name:</b> {index} <br />' for index in trans.index],
                                     marker=dict(opacity=0.4, size=20, color=COLOR[str(v_level)]))
        return grid_plot

    fig = go.Figure()
    try:
        for v in voltage_levels:
            if not nodes.empty:
                fig.add_trace(get_nodes(v))
            if not consumers.empty:
                fig.add_trace(get_consumers(v))
            if not edges.empty:
                fig.add_trace(get_edges(v))
            if not transformers.empty:
                fig.add_trace(get_transformers(v))

        fig.update_layout(mapbox=dict(accesstoken=API_KEY, bearing=0, pitch=0, zoom=16, style='outdoors',
                                      center=go.layout.mapbox.Center(lat=nodes['lat'].mean(), lon=nodes['lon'].mean())),
                          autosize=True)
        return fig

    except Exception as e:
        print(repr(e))
        return fig
