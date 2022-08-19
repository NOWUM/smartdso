from sqlalchemy import create_engine
from matplotlib import pyplot as plt
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import pandas as pd
import os
import numpy as np

DATABASE_URI = os.getenv('DATABASE_URI', 'postgresql://opendata:opendata@10.13.10.41:5432/smartgrid')
ENGINE = create_engine(DATABASE_URI)

SCENARIOS = ['EV100PV100PRC1.0', 'EV100PV100PRC1.7', 'EV100PV100PRC2.7', 'EV100PV100PRC4.0']

FONT = dict(family="Verdana", size=12, color="black")
COLORS = {'EV100PV100PRC1.0': 'rgb(0,204,0)', 'EV100PV100PRC1.7': 'rgb(0,204,153)',
          'EV100PV100PRC2.7': 'rgb(0,102,153)', 'EV100PV100PRC4.0': 'rgb(0,0,153)'}


def get_transformer_utilization(scenario):
    # -> get transformer utilization
    query = f"select time, utilization from grid " \
            f"where scenario='{scenario}' and asset='transformer'"
    utilization = pd.read_sql(query, ENGINE)
    utilization['time'] = utilization['time'].apply(pd.to_datetime)
    return utilization


def get_final_charge(scenario):
    # -> get sum power over each iteration in a scenario
    query = f"select time, 0.25 * sum(final_charge) as demand from cars" \
            f" where scenario='{scenario}' " \
            f"group by time"
    demand = pd.read_sql(query, ENGINE)
    demand['time'] = demand['time'].apply(pd.to_datetime)
    demand['demand'] /= 30
    return demand


def get_charging_at_quarters(scenario):
    # ->
    query = f"select to_char(res.ti, 'hh24:mi') as inter, avg(res.planned) as planned, avg(res.final) as final " \
            f"from (select  time as ti, sum(planned_grid_consumption) as planned, sum(final_grid_consumption) as final " \
            f"from residential where scenario='{scenario}' group by ti) as res " \
            f"group by inter"
    data = pd.read_sql(query, ENGINE)
    return data


def get_prices():
    prices = pd.read_csv(r'./participants/data/2022_prices.csv', index_col=0,
                         parse_dates=True)
    prices = prices.resample('15min').ffill()
    prices['price'] /= 10
    return prices


def get_values_in_price_intervals(d: pd.DataFrame, parameter: str = 'utilization'):

    values, names, counter = [], [], []
    intervals = list(np.linspace(0, 40, 5)) + [1e9]

    for min_, max_ in zip(intervals[:-1], intervals[1:]):
        values += [d.loc[(d['price'] >= min_) & (d['price'] < max_), parameter].values]
        counter += [sum((d['price'] >= min_) & (d['price'] < max_))]
        if max_ == 1e9:
            names += [f'{round(min_, 0)} <']
        else:
            names += [f'{round(min_, 0)} - {round(max_, 0)}']

    return values, names, counter


def create_scatter_box_plot():

    fig = make_subplots(rows=2, cols=1, specs=[[{"secondary_y": False}], [{"secondary_y": False}]],
                        shared_xaxes=True, row_width=[0.1, 0.4])

    values, names, counter = [], [], []

    for SCENARIO in SCENARIOS:
        util = get_transformer_utilization(SCENARIO)
        dem = get_final_charge(SCENARIO)
        prc = get_prices()

        util['price'] = [prc.loc[t, 'price'] for t in util['time'].values]
        data = util.loc[:, ['utilization', 'price']]
        data = data.drop_duplicates()

        values, names, counter = get_values_in_price_intervals(d=data, parameter='utilization')

        for val, name in zip(values, names):
            fig.add_trace(go.Box(y=val, name=name, showlegend=False, marker_color=COLORS[SCENARIO],
                                 boxpoints=False), row=1, col=1)

        dem['price'] = [prc.loc[t, 'price'] for t in dem['time'].values]
        data = dem.loc[:, ['demand', 'price']]
        data = data.drop_duplicates()

        values, names, counter = get_values_in_price_intervals(d=data, parameter='demand')
        fig.add_trace(go.Bar(name=SCENARIO, x=names, y=[sum(val) / 1e3 for val in values],
                             marker_color=COLORS[SCENARIO]), row=2, col=1)

    for n, c in zip(names, counter):
        fig.add_annotation(x=n, y=50, showarrow=False, text=f'Charging Options: {c}', row=2, col=1)

    # -> set axes titles
    fig.update_yaxes(title_text="Utilization [%]",
                     secondary_y=False,
                     showgrid=True,
                     range=[0, 201],
                     gridwidth=0.1,
                     gridcolor='rgba(0, 0, 0, 0.5)',
                     row=1, col=1)
    fig.update_yaxes(title_text="Total Charged [MWh]",
                     secondary_y=False,
                     showgrid=True,
                     gridwidth=0.1,
                     gridcolor='rgba(0, 0, 0, 0.5)',
                     row=2, col=1)
    fig.update_xaxes(title_text="Price [ct/kWh]",
                     showgrid=False,
                     gridwidth=0.1,
                     gridcolor='rgba(0, 0, 0, 0.5)',
                     row=2, col=1)

    fig.update_layout(font=FONT, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                      boxmode='group', boxgroupgap=0.005, boxgap=0.005, barmode='group',
                      bargap=0.005)
    fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=-0.1, xanchor="right", x=1))
    fig.write_html(f'scatter_boxplot_utilization.html')


def create_shifted_charging(all: bool = False):
    if all:
        rows = len(SCENARIOS)
        specs = [[{"secondary_y": True}] for _ in range(rows)]
        scenarios = SCENARIOS
    else:
        rows=1
        specs = [[{"secondary_y": True}]]
        scenarios = [SCENARIOS[-1]]
    prc = get_prices()
    prc = prc.groupby(prc.index.hour).mean()
    prc = np.asarray([4*[p] for p in prc['price'].values]).flatten()

    fig = make_subplots(rows=rows, cols=1, specs=specs, shared_xaxes=True)
    for scenario, row in zip(scenarios, range(1, rows+1)):
        d = get_charging_at_quarters(scenario=scenario)
        fig.add_trace(go.Bar(
            x=d['inter'].values,
            y=d['planned'].values / 1e3,
            name='Planned Charging',
            showlegend=True,
            marker_color='rgb(0,204,153)'
        ), col=1, row=row)
        fig.add_trace(go.Bar(
            x=d['inter'].values,
            y=d['final'].values / 1e3,
            name='Final Charging',
            showlegend=True,
            marker_color='rgb(0,102,153)'
        ), col=1, row=row)
        fig.add_trace(go.Scatter(
            x=d['inter'].values,
            y=prc,
            mode='lines',
            name='DayAhead Price',
            line=dict(color='rgb(204,0,0)')
        ), col=1, row=row, secondary_y=True)


    # Here we modify the tickangle of the xaxis, resulting in rotated labels.
    fig.update_yaxes(title_text="Mean Charging Power [MW]",
                     secondary_y=False,
                     showgrid=True,
                     range=[0, 45],
                     gridwidth=0.1,
                     gridcolor='rgba(0, 0, 0, 0.5)')
    fig.update_yaxes(title_text="Mean Market Price [ct/kWh]",
                     secondary_y=True,
                     showgrid=False,
                     gridwidth=0.1,
                     range=[0, 45],
                     gridcolor='rgba(0, 0, 0, 0.5)')

    # fig.update_layout(barmode='group', xaxis_tickangle=-45)

    fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=-0.1, xanchor="right", x=1))
    fig.update_layout(font=FONT, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                      boxmode='group', barmode='group', bargap=0.005)

    fig.write_html(f'test.html')


def get_price_sensitivities():
    senses = []
    for slope in [1, 1.7, 2.7, 4.0]:
        X = 100/slope
        sens = [50*np.exp(-x/X)/X for x in np.arange(0.1, 100.1, 0.1)]
        # sens -= min(sens)
        senses += [np.asarray(sens)]
    return senses


if __name__ == "__main__":
    pass
    # create_scatter_box_plot()
    # create_shifted_charging()
    price = get_prices()
    price = price.sort_values('price', ascending=False).values.flatten() / 100 + 0.11
    sens = get_price_sensitivities()
    for s in sens:
        plt.plot(np.arange(0.1, 100.1, 0.1), s)

    plt.plot(np.linspace(0.1, 100, len(price)), price, 'black')

    plt.ylabel('Marginal [d€/dSoC]')
    plt.xlabel('SoC [%]')
    plt.legend(['Sense 1', 'Sense 2', 'Sense 3', 'Sense 4',
                'Sorted Price'])

    plt.show()
