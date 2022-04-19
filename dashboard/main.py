from datetime import datetime, date
from urllib.parse import urlparse, parse_qsl, urlencode
import dash
from dash.exceptions import PreventUpdate
from dash.dependencies import Input, Output
from dash import dcc
from dash import html
import plotly.graph_objects as go
import plotly.express as px

app = dash.Dash(__name__, meta_tags=[{"name": "viewport", "content": "width=device-width"}])
server = app.server
app_name = 'Smart DSO'

layout = html.Div(
    [dcc.Location(id='url_app', refresh=False),
     html.Div(
         [
             html.Div(
                 [
                     html.H1(
                         app_name,
                         style={"margin-bottom": "0px"},
                     ),
                     html.H5(
                         "Simulation Results", style={"margin-top": "70px"},
                     ),
                 ],
                 className="eleven columns",
                 id="title",
             ),
             html.Div(
                 [
                     html.Img(
                         src=app.get_asset_url("fh-aachen.png"),
                         id="plotly-image",
                         style={
                             "height": "200px",
                             "width": "60px",
                             "margin-bottom": "0px",
                             "backgroundColor": 'white',
                         },
                     )
                 ],
                 className="one columns",
             ),
         ],
         id="header",
         className="row flex-display pretty_container",
         style={"margin-bottom": "25px"},
     ),
     html.Div(
         [
             html.Div(
                 [
                     html.P(
                         "Select Grid:",
                         className="control_label",
                     ),
                     dcc.Dropdown(id='name_picker',
                                  options=[{'label': x, 'value': x}
                                           for x in [str(i) for i in range(5)]],
                                  value=['1'],
                                  multi=True,
                                  ),
                     html.P(
                         "Select Time Filter:",
                         className="control_label",
                     ),
                     dcc.DatePickerRange(
                         id='datepicker',
                         min_date_allowed=date(2022, 1, 1),
                         max_date_allowed=date(2023, 1, 15),
                         start_date=date(2022, 1, 1),
                         end_date=date(2022, 1, 15),
                         display_format='DD.MM.YY',
                         show_outside_days=True,
                         start_date_placeholder_text='MMMM Y, DD'
                     ),
                     html.P("Aggregation Intervall:",
                            className="control_label"),
                     dcc.RadioItems(
                         id="groupby_control",
                         options=[
                             {"label": "Hour", "value": "hour"},
                             {"label": "1/4 hour", "value": "minute"},
                         ],
                         value="hour",
                         labelStyle={"display": "inline-block"},
                         className="dcc_control",
                     ),
                 ],
                 className="pretty_container three columns",
                 id="cross-filter-options",
             ),
             html.Div(
                 [
                     html.P(
                         "Traffic Map",
                         className="control_label",
                     ),
                     dcc.Graph(id='choro_graph',
                               config={"displaylogo": False}),
                 ],
                 id="locationMapContainer",
                 className="pretty_container nine columns",
             ),
         ],
         className="row flex-display",
     ),
     # empty Div to trigger javascript file for graph resizing
     html.Div(id="output-clientside"),
     html.Div(
         [
             dcc.Graph(
                 id="traffic_graph", config={"displaylogo": False}),
             dcc.Graph(
                 id="traffic_animation", config={"displaylogo": False})
         ]
     ),
     html.Div([
         dcc.Link('Data comes from TomTom API',
                  href='https://developer.tomtom.com/traffic-api/documentation/traffic-flow/flow-segment-data',
                  refresh=True),
         html.Br(),
         dcc.Link('Legal Notice', refresh=True,
                  href='https://demo.nowum.fh-aachen.de/info.html'),
     ],
         className="pretty_container",
     ),
     ])


############ Controls   ##############

@app.callback(
    Output('name_picker', 'value'),
    [Input('choro_graph', 'clickData'),
     Input('url_acdatep', 'href')])
def update_dropdown(clickData, href):
    # zur initialisierung
    if clickData is None:
        if href:
            state = parse_state(href)
            if 'name_picker' in state:
                plants = state['name_picker'].strip(
                    "][").replace("'", '').split(',')
                return plants
    else:
        # print(clickData['points'][0])
        if 'hovertext' in clickData['points'][0]:
            street_name = clickData['points'][0]['hovertext']
            return [street_name]

    return ['A 4']


component_ids = ['start_date', 'end_date', 'groupby_control',
                 'name_picker']


def parse_state(url):
    parse_result = urlparse(url)
    params = parse_qsl(parse_result.query)
    state = dict(params)
    return state


@app.callback([
    Output("datepicker", "start_date"),
    Output("datepicker", "end_date"),
    Output("groupby_control", "value"),
    # Output("name_picker", "value"),
],
    inputs=[Input('url_acdatep', 'href')])
def page_load(href):
    if not href:
        return []
    state = parse_state(href)
    # print(href)
    # for element in elements
    if all(element in state for element in ['start_date', 'end_date', 'groupby_control']):  # , 'name_picker']):
        return state['start_date'], state['end_date'], state['groupby_control']  # , state['name_picker']
    else:
        raise PreventUpdate


@app.callback(Output('url_acdatep', 'search'),
              [
                  Input("datepicker", "start_date"),
                  Input("datepicker", "end_date"),
                  Input("groupby_control", "value"),
                  Input("name_picker", "value")
              ])
def update_url_state(*values):
    state = urlencode(dict(zip(component_ids, values)))
    return f'?{state}'


############ Plant Graph ############
#
#
# @app.callback(
#     Output("traffic_graph", "figure"),
#     [
#         Input("name_picker", "value"),
#         Input("datepicker", "start_date"),
#         Input("datepicker", "end_date"),
#         Input("groupby_control", "value"),
#     ],
# )
# def make_traffic_figure(names, start_date, end_date, group):
#     start = datetime.strptime(start_date, '%Y-%m-%d').date()
#     end = datetime.strptime(end_date, '%Y-%m-%d').date()
#
#     traffic = tt.utilization(Filter(start, end, group), names)
#
#     if traffic.empty:
#         return {'data': [], 'layout': dict(title="No Data Found for current interval")}
#     street_names = list(names)
#     if len(names) < 1:
#         traffic = traffic.groupby('time').mean('utilization')
#         traffic['name'] = 'overall'
#         street_names = ['Overall mean']
#     figure = px.line(traffic, x=traffic.index, y="utilization", color='name', range_y=[0, 1.0])
#     figure.update_layout(title=f"Utilization for {', '.join(street_names)} from {start_date} to {end_date}",
#                          # xaxis_title=group,
#                          yaxis_title='avg Utilization per Interval',
#                          autosize=True,
#                          hovermode="x unified",
#                          legend=dict(font=dict(size=10), orientation="h"), )
#     # figure.update_yaxes(ticksuffix=" utilization")
#     return figure
#
#
# ############ Map   ##############
#
#
# @app.callback(
#     Output('choro_graph', 'figure'),
#     Input("name_picker", "value"))
# def update_map_figure(names):
#     if len(names) < 1:
#         color = 'name'
#     else:
#         locations['selected'] = locations['name'].isin(names)
#         color = 'selected'
#     return px.scatter_mapbox(locations, 'latitude', 'longitude', color=color,
#                              hover_name='name',
#                              height=700, zoom=9,
#                              mapbox_style='carto-positron', center={"lat": 50.8, "lon": 6.4},
#                              title='Aachen city traffic')
#
#
# @app.callback(
#     Output("traffic_animation", "figure"),
#     [
#         Input("name_picker", "value"),
#         Input("datepicker", "start_date"),
#         Input("datepicker", "end_date"),
#         Input("groupby_control", "value"),
#     ],
# )
# def update_traffic_animation(names, start_date, end_date, group):
#     start = datetime.strptime(start_date, '%Y-%m-%d').date()
#     end = datetime.strptime(end_date, '%Y-%m-%d').date()
#
#     traffic = tt.utilizationPoints(Filter(start, end, group))
#     traffic['utilization'] = traffic['utilization'].fillna(0)
#
#     return px.scatter_mapbox(traffic, 'latitude', 'longitude', color='utilization', size='utilization',
#                              hover_name='name',
#                              height=1000, zoom=12,
#                              mapbox_style='carto-positron', center={"lat": 50.778, "lon": 6.084},
#                              title='Aachen city traffic animation',
#                              range_color=[0.0, 1.0],
#                              animation_frame='time', animation_group='latitude')


app.layout = layout

if __name__ == "__main__":
    app.run_server(debug=True, use_reloader=True, host='0.0.0.0', port=8051)
