import streamlit as st
import pydeck as pdk
import pandas as pd
import ssl

# get rid of ssl connection error (certificates)
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

st.header("Map data")
# read in data`
df = pd.read_csv(
    r'https://gist.githubusercontent.com/bafu-DF/f60bd9ac9579d9f830f1f52ce7e79c86/raw/af16f3bb04d5150cc0e139d25d9c706ccbb80215/sampledata.csv',
    sep=',')

layer = pdk.Layer(
    "ScatterplotLayer",
    df,
    pickable=True,
    opacity=0.8,
    filled=True,
    radius_scale=2,
    radius_min_pixels=10,
    radius_max_pixels=500,
    line_width_min_pixels=0.01,
    get_position='[Longitude, Latitude]',
    get_fill_color=[255, 0, 0],
    get_line_color=[0, 0, 0],
)

# Set the viewport location
view_state = pdk.ViewState(latitude=df['Latitude'].iloc[-1], longitude=df['Longitude'].iloc[-1], zoom=13, min_zoom=10,
                           max_zoom=30)

# Render
r = pdk.Deck(layers=[layer], map_style='mapbox://styles/mapbox/satellite-v9',
             initial_view_state=view_state, tooltip={"html": "<b>Point ID: </b> {PointID} <br /> "
                                                             "<b>Longitude: </b> {Longitude} <br /> "
                                                             "<b>Latitude: </b>{Latitude} <br /> "
                                                             "<b> Value: </b>{Value}"})
r

# output of clicked point should be input to a reusable list
selectedID = st.selectbox("Choose point ID", df['PointID'])
