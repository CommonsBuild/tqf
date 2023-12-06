import hvplot.pandas
import numpy as np
import pandas as pd
import panel as pn
import param as pm
from bokeh.models import HoverTool, NumeralTickFormatter
from icecream import ic

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)


def color_based_on_eth_address(val):
    # Use the first 6 characters after '0x' for the color
    hex_color = f'#{val[2:8]}'

    # Convert the hex color to an integer to determine if it's light or dark
    bg_color = int(val[2:8], 16)

    # Determine if background color is light or dark for text color
    text_color = 'black' if bg_color > 0xFFFFFF / 2 else 'white'

    # Return the CSS style with the calculated text color
    return f'background-color: {hex_color}; color: {text_color};'


class Outcomes(pm.Parameterized):

    donations_dashboard = pm.Selector()
    boost_factory = pm.Selector()
    tqf = pm.Selector()

    def view_boosts(self):
        return self.tqf.boosts

    @pm.depends('boost_factory.param', watch=True, on_init=True)
    def boost_tables(self):
        collected_boosts = self.boost_factory.boost_outputs
        contributors = self.donations_dashboard.contributor_dataset()

        boosts = []
        for i, boost in enumerate(self.boost_factory.boosts):
            boost_data = (
                collected_boosts[collected_boosts[f'balance_{i}'] > 0]
                .merge(contributors, how='inner', left_on='address', right_on='voter')
                .drop('address', axis=1)
                .set_index('voter')
            ).reset_index()
            # Format the donations list
            boost_data['donations'] = boost_data['donations'].apply(
                lambda donations: ['${:.2f}'.format(n) for n in donations]
            )

            boosts.append(boost_data)
        return boosts

    def view_boost_tables(self):
        boosts = self.boost_tables()
        charts = []
        for boost in boosts:
            # Use tabulator to display the data
            chart = pn.widgets.Tabulator(
                boost,
                formatters={'donations': {'type': 'textarea', 'textAlign': 'left'}},
            )
            chart.style.applymap(color_based_on_eth_address, subset='voter')

            charts.append(chart)

        return pn.Column(*charts)

    def view_sme_tokens(self):
        # The boost factory output dataset for total boosts
        collected_boosts = self.boost_factory.collect_boosts()

        # Adding SME colors to the token distribution
        charts = []
        for i, boost in enumerate(self.boost_factory.boosts):
            # Chart 1 is the token distribution chart on the boost.
            chart1 = boost.view_signal().opts(tools=[])

            # Chart 2 is inner merge of the contributors dataset and addresses that have positive Boost_i from collected boosts.
            collected_boost = collected_boosts[collected_boosts[f'Boost_{i}'] > 0]
            chart2_data = (
                collected_boost.merge(
                    boost.input.dataset.reset_index(),
                    how='inner',
                    left_on='address',
                    right_on='address',
                )
                .merge(
                    self.donations_dashboard.contributor_dataset(),
                    how='inner',
                    left_on='address',
                    right_on='voter',
                )
                .rename({'index': 'x'}, axis=1)
            )

            chart2_data['color'] = chart2_data['address'].apply(
                lambda x: color_based_on_eth_address(x).split(';')[0].split()[1]
            )

            hover = HoverTool(
                tooltips=[('address', '@address'), ('balance', f'@balance_{i}{{0.00}}')]
            )
            chart2 = chart2_data.rename({'index': 'x'}, axis=1).hvplot.scatter(
                y=f'balance_{i}',
                x='x',
                yformatter=NumeralTickFormatter(format='0,0'),
                alpha=0.8,
                logy=True,
                hover_cols=['address', 'balance'],
                title=self.name,
                tools=[hover],
                size=1000,
                color='color',
                line_color='skyblue',
                xlabel='index',
                shared_axes=False,
            )

            charts.append((chart1 * chart2).opts(shared_axes=False))

        return pn.Row(*charts)

    @pm.depends('tqf.results', watch=True, on_init=True)
    def view_network(self):
        return self.donations_dashboard._contributions_network_view(
            donations_df=self.tqf.boosted_donations,
            donations_column='Boosted Amount',
            funding_outcomes=self.tqf.results,
        )

    @pm.depends('tqf.results', watch=True, on_init=True)
    def view_results_bar(self):
        return self.tqf.view_results_bar()

    def view(self):
        self.donations_dashboard.boost_tables = self.boost_tables
        self.donations_dashboard.sme_list = set.union(
            *[set(boost['voter']) for boost in self.boost_tables()]
        )
        donations_view = self.donations_dashboard.view()
        donations_view[1].append(
            (
                'Subject Matter Expertise',
                pn.Column(
                    # self.view_sme_tokens(),
                    # self.view_boost_tables(),
                ),
            )
        )
        # donations_view[1].active = len(donations_view[1]) - 1

        view = pn.Row(
            pn.Column(
                self.tqf.param['boost_factor'],
                self.tqf.param['matching_pool'],
                self.tqf.param['matching_percentage_cap'],
                self.tqf.param['mechanism'],
            ),
            pn.Column(
                # self.tqf.view_results,
                self.view_network,
                # pn.Row(self.view_network, self.view_results_bar),
            ),
        )
        # Append a layout to the main area, to demonstrate the list-like API
        return view
