import pandas as pd
from matplotlib import pyplot as plt
import matplotlib
import numpy as np
from gridLib import model as gridModel

font = {'family': 'Times New Roman', 'style': 'italic', 'weight': 'medium', 'size': 8}
matplotlib.rc('font', **font)
cm = 1 / 2.54


class EvalPlotter:

    def __init__(self, width: float = 9.05, offset_days=1):
        self.colors = {'pv_generation': (230 / 255, 230 / 255, 74 / 255, 0.65),
                       'availability': (0, 0, 0, 0.6),
                       'market_price': (255 / 255, 0, 0, 0.7),
                       'charging': (255 / 255, 0, 0, 0.7),
                       'usage': (0, 0, 0, 0.2),
                       'soc': (20 / 255, 35 / 255, 90 / 255, 1.0),
                       'used_pv_generation': (230 / 255, 230 / 255, 74 / 255, 0.65)
                       }
        self.pv_colors = {'PV25': (1, 204 / 255, 153 / 255),
                          'PV50': (1, 102 / 255, 102 / 255),
                          'PV100': (1, 1, 51 / 255),
                          'PV80': (1, 0.5, 0 / 255)}

        self.sub_colors = {0: (1, 0, 0), 1: (1, 0.5, 0), 2: (204 / 255, 204 / 255, 0),
                           3: (0.5, 1, 0), 4: (0, 153 / 255, 0), 5: (51 / 255, 1, 153 / 255),
                           6: (0, 1, 1), 7: (0, 0.5, 1), 8: (0, 0, 1),
                           9: (0.5, 0.5, 0.5)}

        self._grid_model = gridModel.GridModel()
        self._all_lines = gridModel.total_edges
        self._all_nodes = gridModel.total_nodes

        self._width = width
        self._offset = offset_days
        self._s1 = 96 * self._offset
        self._s2 = 96 * (self._offset + 7)

    def plot_charging_compare(self, data: dict, car_id: str):
        _, pv, kwh = car_id.split('_')

        plt_data = plt.subplots(len(data), 1, sharex='all', figsize=(self._width * cm, 1.5 * self._width * cm), dpi=300)
        fig, plots = plt_data[0], plt_data[1:]
        plot, sec_plot = plots[0][0], None
        fig.suptitle(f'Installed PV: {pv} kW, Battery Capacity: {kwh} kWh')
        for plot, values in zip(plots[0], data.items()):
            scenario, val = values[0], values[1]
            val = val.loc[val.index[self._s1:self._s2], :]
            plot.set_title(scenario, fontsize=6)
            plot.set_ylabel('charging [kW]')
            plot.plot(val.index, val['charging'].values, color=self.colors['charging'], linewidth=1)
            plot.fill_between(val.index, val['used_pv_generation'].values,
                              color=self.colors['used_pv_generation'], linewidth=0)

            sec_plot = plot.twinx()
            sec_plot.set_ylabel('SoC / Usage [%]')
            sec_plot.plot(val.index, val['soc'].values * 100, color=self.colors['soc'], linewidth=1)
            sec_plot.fill_between(val.index, 100 * val['usage'].values, color=self.colors['usage'], linewidth=0.25)

            major_ticks = [val.index[i] for i in range(0, len(val.index), 3 * 96)]
            minor_ticks = [val.index[i] for i in range(0, len(val.index), 1 * 96)]
            plot.xaxis.set_ticks(major_ticks)
            plot.xaxis.set_ticks(minor_ticks, minor=True)
            plot.grid(True)

        if plot is not None and sec_plot is not None:
            plot.set_xlabel('time')
            plot.legend(['Total Charging', 'PV Charging'], loc=9, bbox_to_anchor=(0.5, -0.4),
                        fancybox=True, ncol=2, frameon=False)
            sec_plot.legend(['State of Charge', 'Car Usage'], loc=9, bbox_to_anchor=(0.5, -0.65),
                            fancybox=True, ncol=2, frameon=False)
        fig.tight_layout()

        return fig

    def plot_impact(self, data: dict, pv_colors: bool = False, linewidth: float = 0.5):
        fig, (ax, ax_zoom) = plt.subplots(1, 2, figsize=(self._width * cm, self._width / 2 * cm), dpi=300,
                                          gridspec_kw={'width_ratios': [2, 1]})

        num = len([*data.values()][0])
        # drawing the same order per plot, the colors are the same in both plots
        for name, val in data.items():
            if pv_colors:
                pv = name.split(' ')[-1]
                ax.plot(100 * np.arange(num) / num, val.values, linewidth=linewidth, color=self.pv_colors[pv])
                val = val[:int(len(val) * 0.2)]
                ax_zoom.plot(100 * np.arange(len(val)) / num, val.values, linewidth=linewidth, color=self.pv_colors[pv])
            else:
                ax.plot(100 * np.arange(num) / num, val.values, linewidth=linewidth)
                val = val[:int(len(val) * 0.2)]
                ax_zoom.plot(100 * np.arange(len(val)) / num, val.values, linewidth=linewidth)

        # add scatters afterwards so that they don't appear in first 4 items of legend
        for name, val in data.items():
            p99_quantile = val.quantile(0.99)
            if pv_colors:
                pv = name.split(' ')[-1]
                ax_zoom.scatter([1], p99_quantile, marker='o', s=1, color=self.pv_colors[pv])
                ax.scatter([1], p99_quantile, marker='o', s=1, color=self.pv_colors[pv])
            else:
                ax_zoom.scatter([1], p99_quantile, marker='o', s=1,)
                ax.scatter([1], p99_quantile, marker='o', s=1,)

        ax.yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())
        ax.grid(which='both', linewidth=0.25)
        ax.set_ylabel('Utilization [%]')
        ax.set_xlabel('Num. of Values [%]')
        ax_zoom.yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())
        ax_zoom.grid(which='both', linewidth=0.25)
        # ax_zoom.set_ylabel('Utilization [%]')
        ax_zoom.set_xlabel('Num. of Values [%]')
        # ax.tick_params(which='minor', length=10)
        # legend prints only first 4 lines
        fig.legend([*data.keys()], bbox_to_anchor=(0.55, -0.15), loc='lower center', fontsize=6,
                   fancybox=True, ncol=2, frameon=False)

        fig.tight_layout()

        return fig

    def plot_impact_grid_level(self, transformer_data: dict, total_charged: dict, pv_ratio: dict):
        data = pd.DataFrame.from_dict(pv_ratio, orient='index').sort_index()
        pv_ratio = data.values

        fig, (ax_grid, ax_transformer) = plt.subplots(2, 1, sharex='all',
                                                      gridspec_kw={'height_ratios': [2, 8]},
                                                      figsize=(self._width * cm, self._width * cm), dpi=300)

        grids = [*transformer_data.values()][0].columns
        ax_grid.plot([int(i) for i in data.index], pv_ratio[:, 0], linewidth=0.5, color='black')
        ax_grid.plot([int(i) for i in data.index], pv_ratio[:, 1], linewidth=0.5, color='blue')
        ax_grid.grid(True)
        ax_grid.set_title('Total PV [kW] and Number of Cars', fontsize=6)
        ax_grid.legend(['total kWp', 'no. cars'], loc='lower center', fontsize=6, ncol=2, frameon=False)

        marker = ['^', 'o', 's', 'D']

        scenario_keys = list(transformer_data.keys())

        for grid in grids:
            trans_values = [df[grid].max() for df in transformer_data.values()]
            for i in range(len(trans_values)):
                ax_transformer.scatter(grid, trans_values[i], marker=marker[i], color=self.sub_colors[grid], s=4)

        ax_transformer.grid(True)
        ax_transformer.set_ylabel('Transformer Util. [%]')
        ax_transformer.set_xticks(range(10))

        ax_transformer.set_ylim([10, 100])
        ax_transformer.legend(scenario_keys, bbox_to_anchor=(0.55, -0.3), loc='lower center', fontsize=6, ncol=2,
                              frameon=False)

        fig.tight_layout()

        return fig

    def plot_pv_impact_grid_level(self, transformer_data: dict, total_charged: dict, pv_ratio: dict):
        data = pd.DataFrame.from_dict(pv_ratio, orient='index').sort_index()
        pv_ratio = data.values

        fig, (ax_grid, ax_25, ax_transformer) = plt.subplots(3, 1, sharex='all',
                                                             gridspec_kw={'height_ratios': [2, 2, 6]},
                                                             figsize=(self._width * cm, self._width * cm), dpi=300)

        grids = [*transformer_data.values()][0].columns
        ax_grid.plot([int(i) for i in data.index], pv_ratio[:, 0], linewidth=0.5, color='black')
        ax_grid.plot([int(i) for i in data.index], pv_ratio[:, 1], linewidth=0.5, color='blue')
        ax_grid.grid(True)
        ax_grid.set_title('Total PV [kW] and Number of Cars', fontsize=6)
        ax_grid.legend(['total kWp', 'no. cars'], loc='lower center', fontsize=6, ncol=2, frameon=False)

        scenario_keys = list(transformer_data.keys())
        scenario_keys.remove('Case A PV25')
        utilization_values = transformer_data['Case A PV25']
        max_utilization_values = {}
        for grid in grids:
            q95 = utilization_values[grid].quantile(0.95)
            max_utilization = utilization_values.loc[utilization_values.values > q95, grid].mean()
            max_utilization_values[grid] = max_utilization
            ax_25.scatter(grid, max_utilization, marker='^', color=self.sub_colors[grid], s=4)

        for grid in grids:
            marker = (m for m in ['o', 's', 'D'])
            for sc, df in transformer_data.items():
                if 'PV25' in sc:
                    continue
                else:
                    q95 = df[grid].quantile(0.95)
                    max_utilization = df.loc[df.values > q95, grid].mean()
                    delta = max_utilization_values[grid] - max_utilization
                    ax_transformer.scatter(grid, delta, marker=next(marker), color=self.sub_colors[grid], s=4)

        ax_25.grid(True)
        ax_25.set_ylabel('Util. [%]')
        ax_transformer.grid(True)
        ax_transformer.set_xticks(range(10))
        ax_transformer.set_ylabel('\u0394Util. [%]')

        ax_transformer.legend(scenario_keys, bbox_to_anchor=(0.55, -0.4), loc='lower center', fontsize=6, ncol=2,
                              frameon=False)

        fig.tight_layout()

        return fig

    def plot_grid(self):
        fig, ax = plt.subplots(1, 1, sharex='all', figsize=(self._width * cm, self._width * cm), dpi=300)

        self._all_lines['sub_grid'] = 0
        self._all_nodes['sub_grid'] = 0
        for name, network in self._grid_model.sub_networks.items():
            m = network['model']
            lines = m.lines.index
            nodes = m.buses.index
            self._all_lines.loc[lines, 'sub_grid'] = int(name)
            self._all_nodes.loc[nodes, 'sub_grid'] = int(name)

        for grid in self._all_nodes['sub_grid'].unique():
            nodes = self._all_nodes.loc[self._all_nodes['sub_grid'] == grid]
            ax.scatter(nodes['lon'].values, nodes['lat'].values, color=self.sub_colors[grid], s=5)
        for grid in self._all_nodes['sub_grid'].unique():
            lines = self._all_lines.loc[self._all_lines['sub_grid'] == grid]
            for idx, values in lines.iterrows():
                ax.plot(eval(values.lon_coords), eval(values.lat_coords), color='black', linewidth=1)

        ax.set_xticks([])
        ax.set_yticks([])

        ax.legend([f'Id: {g}' for g in self._all_nodes['sub_grid'].unique()], loc=4)

        fig.tight_layout()

        return fig

    def plot_utilization(self, data: dict, date_range: list, sub_id: int = 5):

        plt_data = plt.subplots(len(data), 1, sharex='all', figsize=(self._width * cm, 1.5 * self._width * cm), dpi=300)
        fig, plots = plt_data[0], plt_data[1:]
        date_range = date_range[self._s1:self._s2]

        fig.suptitle(f'Utilization Transformer Subgrid {sub_id}')

        for plot, values in zip(plots[0], data.items()):
            scenario, val = values[0], values[1]
            plot.set_title(scenario, fontsize=6)
            plot.set_ylabel('Utilization [%]')
            x = date_range
            y1 = val[0][self._s1:self._s2]
            y2 = val[1][self._s1:self._s2]
            y_sim = val[2][self._s1:self._s2]
            plot.plot(x, y1, color=self.colors['soc'], linewidth=0.5)
            plot.plot(x, y2, color=self.colors['soc'], linewidth=0.5)
            plot.plot(x, y_sim, color=self.colors['market_price'], linewidth=0.5)
            plot.fill_between(x=x, y1=y1, y2=y2, color='grey', alpha=0.5, linewidth=0)

            major_ticks = [date_range[i] for i in range(0, len(date_range), 3 * 96)]
            minor_ticks = [date_range[i] for i in range(0, len(date_range), 1 * 96)]
            plot.xaxis.set_ticks(major_ticks)
            plot.xaxis.set_ticks(minor_ticks, minor=True)
            plot.grid(True)
            plot.set_ylim([0, 100])

        fig.tight_layout()

        return fig

    def plot_overview(self, data: pd.DataFrame):
        fig, ax = plt.subplots(1, 1, figsize=(self._width * cm, 0.5 * self._width * cm), dpi=300)
        # data = data.resample('h').mean()
        data = data.iloc[self._s1:self._s2, :]
        data.loc[:, 'pv'] /= data['pv'].max()
        data.loc[:, 'pv'] *= 100

        ax.plot(data.index, data['price'].values,
                color=self.colors['market_price'], linewidth=1)

        sec_ax = ax.twinx()
        sec_ax.fill_between(x=data.index, y1=data['pv'].values,
                            color=self.colors['used_pv_generation'], linewidth=1)

        sec_ax.plot(data.index, data['availability'].values,
                    color=self.colors['availability'], linewidth=1)

        ax.set_xlabel('time')
        ax.set_ylabel('Market Price [ct/kWh]')
        sec_ax.set_ylabel('Availability / PV Feed-In [%]')

        ax.grid(True)
        major_ticks = [data.index[i] for i in range(0, len(data.index), 3 * 96)]
        minor_ticks = [data.index[i] for i in range(0, len(data.index), 1 * 96)]
        ax.xaxis.set_ticks(major_ticks)
        ax.xaxis.set_ticks(minor_ticks, minor=True)

        fig.tight_layout()

        return fig

    def plot_benefit_function(self, data: pd.DataFrame):
        fig, ax = plt.subplots(1, 1, figsize=(self._width * cm, self._width * cm), dpi=300)
        ax.grid(True)
        ax.set_xlabel('SoC [%]')
        ax.set_ylabel('Price [ct/kWh]')
        for col in data.columns:
            ax.plot(data.index.values, data[col].values, linewidth=0.75)

        fig.tight_layout()

        return fig

    def plot_grid_fee_function(self, data: dict):
        fig, ax = plt.subplots(1, 1, figsize=(self._width * cm, 0.5 * self._width * cm), dpi=300)
        ax.grid(True)
        ax.set_xlabel('Utilization [%]')
        ax.set_ylabel('Price [ct/kWh]')
        util = [*data.keys()]
        price = [*data.values()]
        ax.plot(util, price, linewidth=0.75)
        fig.tight_layout()

        return fig
