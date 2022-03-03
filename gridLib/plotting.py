import plotly.graph_objs as go
from plotly.offline import plot
import numpy as np
import pandas as pd

def show_plot(nodes: pd.DataFrame,
              edges: pd.DataFrame,
              transformers: pd.DataFrame,
              consumers:pd.DataFrame,
              layers=None, voltage_levels:list = [0.4, 10, 20, 110]):
    """
    Plot the resulting grid model with a plotly graphic object
    The figure displays each voltage levels in an own trace containing nodes, lines and transformers
    The figure is shown in the browser based on an html file, which is created in the method

    Parameters
    ----------
    :return:
    """

    color = {'0.4': '#aa5555', '10': '#278511', '20': '#222233', '110': '#424949'}
    sizes = {'0.4': 6, '10': 7, '20': 9, '110': 10}

    def get_edges(voltage_level=0.4):
        lons, lats, names = [], [], []
        lines = edges[edges['v_nom'] == voltage_level]
        for name, attributes in lines.iterrows():
            shape = attributes['shape']
            x, y = shape.xy
            lats = np.append(lats, y)
            lons = np.append(lons, x)
            names = np.append(names, [f'<b>Name:</b> {name} <br />'
                                      f'<b>S:</b> {round(attributes["s_nom"], 2)} MVA'] * len(y))
            lats = np.append(lats, [None])
            lons = np.append(lons, [None])
            names = np.append(names, [None])

        grid_plot = go.Scattermapbox(name=f'lines on {voltage_level} kV',
                                     lon=lons, lat=lats, hoverinfo='text',
                                     hovertext=names, mode='lines',
                                     line=dict(width=2, color=color[str(voltage_level)]))
        return grid_plot

    def get_nodes(voltage_level):
        busses = nodes.loc[nodes['v_nom'] == voltage_level]
        grid_plot = go.Scattermapbox(name=f'Nodes on {voltage_level} kV', mode='markers',
                                     lon=busses['lon'], lat=busses['lat'],
                                     hoverinfo='text', hovertext=[f'<b>Name:</b> {index}' for index in busses.index],
                                     marker=dict(size=sizes[str(voltage_level)], color=color[str(voltage_level)]))
        return grid_plot

    def get_consumers(voltage_level):
        demand = consumers.loc[consumers['v_nom'] == voltage_level]
        grid_plot = go.Scattermapbox(name=f'Consumers on {voltage_level} kV', mode='markers',
                                     lon=demand['lon'], lat=demand['lat'],
                                     hoverinfo='text', hovertext=[f'<b>Name:</b> {index}' for index in demand.index],
                                     marker=dict(symbol='zoo'))
        return grid_plot

    def get_transformers(voltage_level):
        trans = transformers.loc[transformers['v0'] == voltage_level]
        grid_plot = go.Scattermapbox(name=f'Transformers on {voltage_level} kV (low_level)', mode='markers',
                                     lon=trans['lon'], lat=trans['lat'], hoverinfo='text',
                                     hovertext=[f'<b>Name:</b> {index}, <br />'
                                                f'<b>V1:</b> {trans.loc[index, "v0"]} kV <br />'
                                                f'<b>V2:</b> {trans.loc[index, "v1"]} kV <br />'
                                                f'<b>S:</b> {trans.loc[index, "s_nom"]} MVA'
                                                for index in trans.index],
                                     marker=dict(opacity=0.4, size=20, color=color[str(voltage_level)]))
        return grid_plot

    fig = go.Figure()
    try:
        for voltage_level in voltage_levels:
            fig.add_trace(get_nodes(voltage_level))
            fig.add_trace(get_consumers(voltage_level))
            fig.add_trace(get_edges(voltage_level))
            fig.add_trace(get_transformers(voltage_level))


        api_key = 'pk.eyJ1Ijoicmlla2VjaCIsImEiOiJjazRiYTdndXkwYnN3M2xteGN2MHhtZjB0In0.33tSDK45TXF3lb3-G147jw'

        fig.update_layout(mapbox=dict(accesstoken=api_key, bearing=0, pitch=0, zoom=13,
                                      center=go.layout.mapbox.Center(lat=50.805063, lon=6.49778),
                                      style='satellite', layers=layers), autosize=True)
        plot(fig, 'temp-plot.html')

    except Exception as e:
        print(repr(e))