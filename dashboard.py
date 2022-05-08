import pandas as pd
import streamlit as st
import logging
from datetime import timedelta

from gridLib.model import GridModel
from interfaces.results import Results
from interfaces.simulation import update_image, initialize_scenario, start_scenario
from plotLib.plot import plot_grid, plot_charge, plot_car_usage, plot_utilization

# -> logging information
logger = logging.getLogger('Control Center')
# -> grid reference
grid = GridModel()
# -> simulation servers
servers = ["10.13.10.54", "10.13.10.55", "10.13.10.56"]
# -> scenarios in DB
r = Results()


# -> start a new simulation
def run_simulation(s, ev_ratio, charge_limit, sd, ed, df):
    r.delete_scenario(scenario=s)
    with st.spinner('Update Image...'):
        update_image(s)
        logger.info('updated images')
    with st.spinner('Build Scenario...'):
        initialize_scenario(s, ev_ratio=ev_ratio, minimum_soc=charge_limit, start_date=sd, end_date=ed, dynamic_fee=df)
        logger.info('built scenario')
    with st.spinner('Stating Simulation...'):
        start_scenario(s)
        logger.info('started simulation')


# -> Dashboard <-
st.set_page_config(layout="wide")
title = st.title('Smart DSO Dashboard')

# -> scenario selection
with st.sidebar.expander('Select Result Set', expanded=True):
    scenario = st.radio("Select Scenario:", r.scenarios, key='charging_scenario')
    r.scenario = scenario

try:
    # -> get simulation results
    sim_result = r.get_vars()
    charged = sim_result.loc[:, 'charged']
    shifted = sim_result.loc[:, ['avg_shifted', 'min_shifted', 'max_shifted']]
    shifted.columns = map(lambda x: x.split('_')[0], shifted.columns)
    price = sim_result.loc[:, ['avg_price', 'min_price', 'max_price']]
    price.columns = map(lambda x: x.split('_')[0], shifted.columns)
except:
    pass

with st.sidebar.expander('Configure Simulation', expanded=False):
    with st.form(key='simulation_vars'):
        # -> ev ratio
        slider_ev = st.slider("EV-Ratio", min_value=0, max_value=100, value=50, step=1)
        # -> charging limit
        slider_charge = st.slider("Charging-Limit", min_value=-1, max_value=100, value=50)
        # -> no dynamic greed fees
        dynamic_fee = st.checkbox('Dynamic Grid Fee', value=True)
        # -> days
        s_date = st.date_input('Start Date', value=pd.to_datetime('2022-01-01'))
        e_date = st.date_input('End Date', value=pd.to_datetime('2022-01-15'))
        # -> simulation server
        sim_server = st.selectbox("Start Simulation on Server", servers)
        # -> submit button
        run = st.form_submit_button("Run Simulation")

        if run:
            run_simulation(sim_server, slider_ev, slider_charge, s_date, e_date, dynamic_fee)
            st.markdown(f"Start Simulation on Server **{sim_server}**")

# -> charge overview
with st.expander('Charging Overview', expanded=True):
    try:
        st.subheader('Charging- & Price-Overview')
        fig = plot_charge(charged, shifted, price)
        st.plotly_chart(fig, use_container_width=True)

        col1, col2, col3, _ = st.columns([1, 1, 1, 1])
        c = round(charged.values.sum() / 60, 2)
        s = round(shifted['avg'].values.sum() / 60, 2)
        col1.metric("Charged [kWh]", f'{c}')
        col2.metric("Shifted [kWh]", f'{s}')
        col3.metric("Ratio [%]", f'{round(s / c * 100, 2)}')
    except:
        st.write('Error - No Data Found')

# -> car overview
with st.expander('Car', expanded=False):
    try:
        st.subheader('Example Cars & Metrics')
        plot, select = st.columns([3, 1])

        with select:
            iteration = st.selectbox("Select Car:", r.iterations)
        with plot:
            car_result = r.get_cars()
            st.plotly_chart(plot_car_usage(car_result), use_container_width=True)

        evs_result = r.get_evs()
        count_ev, avg_distance, avg_demand, _ = st.columns([1, 1, 1, 1])
        count_ev.metric("Total EVs", f'{int(evs_result.total_ev)}')
        avg_distance.metric("Distance [km/a]", f'{round(365 * evs_result.avg_distance, 2)}')
        avg_demand.metric("Mean Demand [kWh/100km]", f'{round(evs_result.avg_demand, 2)}')
    except:
        st.write('Error - No Data Found')

# -> grid overview
with st.expander('Grid', expanded=False):
    st.subheader('Grid Utilization')
    try:
        plot, tables = st.columns([2, 1])
        with plot:
            st.write(plot_grid(line_utilization=None, sub_id='total'))
        with tables:
            st.caption('Out Line Utilization')
            st.dataframe(r.get_maximal_util(asset='inlet'))
            st.caption('In Line Utilization')
            st.dataframe(r.get_maximal_util(asset='outlet'))
            st.caption('Transformer Utilization')
            st.dataframe(r.get_maximal_util(asset='transformer'))
        st.caption('Lines')
        st.plotly_chart(plot_utilization(r.get_asset_type_util(asset='line')), use_container_width=True)
        st.caption('Transformers')
        st.plotly_chart(plot_utilization(r.get_asset_type_util(asset='transformer')), use_container_width=True)
    except:
        st.write('Error - No Data Found')
