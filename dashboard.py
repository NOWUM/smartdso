import pandas as pd
import streamlit as st
import logging
import os

from gridLib.model import GridModel
from interfaces.simulation import update_image, start_scenario, initialize_scenario, \
    get_data, get_scenarios, get_iterations, get_car
from interfaces.plotting import plot_charging, plot_car, plot_grid


logger = logging.getLogger('Control Center')
grid = GridModel()
servers = ["10.13.10.54", "10.13.10.55", "10.13.10.56"]

scenarios = get_scenarios()


def run_simulation(s, ev_ratio, charge_limit, sd, ed):
    with st.spinner('Update Image...'):
        update_image(s)
    with st.spinner('Build Scenario...'):
        initialize_scenario(s, ev_ratio=ev_ratio, minimum_soc=charge_limit, start_date=sd, end_date=ed)
    with st.spinner('Stating Simulation...'):
        start_scenario(s)


st.set_page_config(layout="wide")
title = st.title('Smart DSO Dashboard')
sim_control = st.sidebar.form("sim_control")

with sim_control:
    slider_ev = st.slider("EV-Ratio", min_value=10, max_value=100, value=50)
    slider_charge = st.slider("Charging-Limit", min_value=-1, max_value=100, value=50)

    s_date = st.date_input('Start Date', value=pd.to_datetime('2022-01-01'))
    e_date = st.date_input('End Date', value=pd.to_datetime('2022-01-15'))

    sim_server = st.selectbox("Start Simulation on Server", servers)
    run = st.form_submit_button("Run Simulation")

    if run:
        run_simulation(sim_server, slider_ev, slider_charge, s_date, e_date)
        str_ = f"Start Simulation on Server **{sim_server}**"
        st.markdown(str_)


with st.expander('Charging', expanded=True):
    st.subheader('Charging & Price Overview')
    plot, select = st.columns([3, 1])
    with select:
        scenario = st.radio("Select Scenario:", scenarios, key='charging_scenario')
    with plot:
        charged = get_data(type_='charged', scenario=scenario)
        shifted = get_data(type_='shifted', scenario=scenario)
        price = get_data(type_='price', scenario=scenario)
        st.plotly_chart(plot_charging(charged, shifted, price), use_container_width=True)
    with st.container():
        col1, col2, col3, _ = st.columns([1, 1, 1, 1])
        c = round(charged.values.sum() / 60, 2)
        s = round(shifted.values.sum() / 60, 2)
        col1.metric("Charged", f'{c} kWh')
        col2.metric("Shifted", f'{s} kWh')
        col3.metric("Ratio", f'{round(s/c * 100, 2)} %')


with st.expander('Car', expanded=False):
    st.subheader('Example Cars & Metrics')
    plot, select = st.columns([3, 1])
    with select:
        scenario = st.radio("Select Scenario:", scenarios, key='car_scenario')
        iteration = st.selectbox("Select Car:", get_iterations(scenario), key='car_iteration')
    with plot:
        car = get_car(scenario, iteration)
        st.plotly_chart(plot_car(car), use_container_width=True)

with st.expander('Grid', expanded=False):
    st.subheader('Grid Utilization')
    plot, select = st.columns([3, 1])
    with select:
        sub_id = st.selectbox("Select Grid:", ['total'] + [f'{i}' for i in range(5)], key='grid_id')
        scenario = st.radio("Select Scenario:", scenarios, key='grid')
    with plot:
        st.write(plot_grid(sub_id))

