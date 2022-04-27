import pandas as pd
import streamlit as st
import logging


from gridLib.model import GridModel
from interfaces.results import get_simulation_results, get_car_usage, get_scenarios, get_iterations, get_lines, \
    get_transformers, get_transformer_utilization
from interfaces.simulation import update_image, initialize_scenario, start_scenario
from plotLib.plot import plot_grid, plot_charge, plot_car_usage, plot_transformer

# -> logging information
logger = logging.getLogger('Control Center')
# -> grid reference
grid = GridModel()
# -> simulation servers
servers = ["10.13.10.54", "10.13.10.55", "10.13.10.56"]
# -> scenarios in DB
scenarios = get_scenarios()


# -> start a new simulation
def run_simulation(s, ev_ratio, charge_limit, sd, ed):
    with st.spinner('Update Image...'):
        update_image(s)
        logger.info('updated images')
    with st.spinner('Build Scenario...'):
        initialize_scenario(s, ev_ratio=ev_ratio, minimum_soc=charge_limit, start_date=sd, end_date=ed)
        logger.info('built scenario')
    with st.spinner('Stating Simulation...'):
        start_scenario(s)
        logger.info('started simulation')


# -> Dashboard <-
st.set_page_config(layout="wide")
title = st.title('Smart DSO Dashboard')


with st.sidebar.expander('Select Result Set', expanded=True):
    # -> scenario selection
    scenario = st.radio("Select Scenario:", scenarios, key='charging_scenario')

with st.sidebar.expander('Configure Simulation', expanded=False):
    with st.form(key='simulation_vars'):
        # -> ev ratio
        slider_ev = st.slider("EV-Ratio", min_value=10, max_value=100, value=50, step=10)
        # -> charging limit
        slider_charge = st.slider("Charging-Limit", min_value=-1, max_value=100, value=50)
        # -> days
        s_date = st.date_input('Start Date', value=pd.to_datetime('2022-01-01'))
        e_date = st.date_input('End Date', value=pd.to_datetime('2022-01-15'))
        # -> simulation server
        sim_server = st.selectbox("Start Simulation on Server", servers)
        # -> submit button
        run = st.form_submit_button("Run Simulation")

        if run:
            run_simulation(sim_server, slider_ev, slider_charge, s_date, e_date)
            st.markdown(f"Start Simulation on Server **{sim_server}**")

# -> charge overview
with st.expander('Charging Overview', expanded=True):
    st.subheader('Charging- & Price-Overview')

    charged = get_simulation_results(type_='charged', scenario=scenario, iteration='total', aggregate='avg')
    min_shifted = get_simulation_results(type_='shifted', scenario=scenario, iteration='total', aggregate='min')
    max_shifted = get_simulation_results(type_='shifted', scenario=scenario, iteration='total', aggregate='max')
    avg_shifted = get_simulation_results(type_='shifted', scenario=scenario, iteration='total', aggregate='avg')
    shifted = pd.DataFrame(data=dict(min=min_shifted['shifted'].values, max=max_shifted['shifted'].values,
                                     avg=avg_shifted['shifted'].values), index=avg_shifted.index)

    # price = get_simulation_results(type_='price', scenario=scenario, iteration='total', aggregate='avg')

    min_price = get_simulation_results(type_='price', scenario=scenario, iteration='total', aggregate='min')
    max_price = get_simulation_results(type_='price', scenario=scenario, iteration='total', aggregate='max')
    avg_price = get_simulation_results(type_='price', scenario=scenario, iteration='total', aggregate='avg')
    price = pd.DataFrame(data=dict(min=min_price['price'].values, max=max_price['price'].values,
                                   avg=avg_price['price'].values), index=avg_shifted.index)

    st.plotly_chart(plot_charge(charged, shifted, price), use_container_width=True)

    col1, col2, col3, _ = st.columns([1, 1, 1, 1])
    c = round(charged.values.sum() / 60, 2)
    s = round(shifted.values.sum() / 60, 2)
    col1.metric("Charged [kWh]", f'{c}')
    col2.metric("Shifted [kWh]", f'{s}')
    col3.metric("Ratio [%]", f'{round(s/c * 100, 2)}')

with st.expander('Car', expanded=False):
    st.subheader('Example Cars & Metrics')
    plot, select = st.columns([3, 1])

    with select:
        iteration = st.selectbox("Select Car:", get_iterations(scenario))
    with plot:
        car, evs = get_car_usage(scenario, iteration)
        st.plotly_chart(plot_car_usage(car), use_container_width=True)

    col1, col2, col3, _ = st.columns([1, 1, 1, 1])
    col1.metric("Total EVs", f'{int(evs["evs"].values[0])}')
    col2.metric("Distance [km/a]", f'{round(365*evs["distance"].values[0], 2)}')
    col3.metric("Mean Demand [kWh/100km]", f'{round(evs["demand"].values[0], 2)}')


with st.expander('Grid', expanded=False):
    st.subheader('Grid Utilization')

    sub_id = st.selectbox("Select Grid:", ['total'] + [f'{i}' for i in range(5)], key='grid_id')
    show_utilization = st.checkbox("Show Maximal Utilization", value=False)
    max_lines, total_lines = get_lines(scenario, sub_id)
    max_transformers = get_transformers(scenario, sub_id)

    if show_utilization:
        plt = plot_grid(line_utilization=total_lines, sub_id=sub_id)
    else:
        plt = plot_grid(line_utilization=None, sub_id=sub_id)

    col1, col2 = st.columns([1, 1])
    with col1:
        st.write(plt)
    with col2:
        st.caption('Maximal Line Utilization')
        st.dataframe(max_lines)
        st.caption('Maximal Transformer Utilization')
        st.dataframe(max_transformers)

    if sub_id != 'total':
        transformer_utilization = get_transformer_utilization(scenario=scenario, sub_id=sub_id)
        st.plotly_chart(plot_transformer(transformer_utilization), use_container_width=True)


