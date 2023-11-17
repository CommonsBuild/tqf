import hvplot.pandas
import numpy as np
import pandas as pd
import panel as pn
import param as pm
from bokeh.models import HoverTool, NumeralTickFormatter


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

    def view_sme(self):
        collected_boosts = self.boost_factory.collect_boosts()
        contributors = self.donations_dashboard.contributor_dataset()

        charts = []
        for i, boost in enumerate(self.boost_factory.boosts):
            chart_data = (
                collected_boosts[collected_boosts[f'balance_{i}'] > 0]
                .merge(contributors, how='inner', left_on='address', right_on='voter')
                .drop('address', axis=1)
                .set_index('voter')
            ).reset_index()
            # Format the donations list
            chart_data['donations'] = chart_data['donations'].apply(
                lambda donations: ['${:.2f}'.format(n) for n in donations]
            )

            # Use tabulator to display the data
            chart = pn.widgets.Tabulator(
                chart_data,
                formatters={'donations': {'type': 'textarea', 'textAlign': 'left'}},
            )
            chart.style.applymap(color_based_on_eth_address, subset='voter')

            charts.append(chart)

        return pn.Column(*charts)

    def view_sme_tokens(self):
        collected_boosts = self.boost_factory.collect_boosts()
        contributors = self.donations_dashboard.contributor_dataset()

        charts = []
        for i, boost in enumerate(self.boost_factory.boosts):
            chart_data = (
                collected_boosts[collected_boosts[f'balance_{i}'] > 0]
                .merge(
                    contributors.reset_index(),
                    how='inner',
                    left_on='address',
                    right_on='voter',
                )
                .drop('address', axis=1)
            )

            chart_data['color'] = chart_data['voter'].apply(
                lambda x: color_based_on_eth_address(x).split(';')[0].split()[1]
            )

            chart1 = boost.view_signal()
            hover = HoverTool(
                tooltips=[('address', '@address'), ('balance', '@balance{0.00}')]
            )
            # chart2 = chart_data.hvplot.scatter(
            #     y='balance',
            #     x='index',
            #     yformatter=NumeralTickFormatter(format='0,0'),
            #     alpha=0.8,
            #     logy=True,
            #     hover_cols=['address', 'balance'],
            #     title=self.name,
            #     tools=[hover],
            #     size=200,
            #     color='color',
            #     line_color='skyblue',
            #     xlabel='index',
            #     shared_axes=False,
            # )

            charts.append(chart1)  # * chart2)

        return pn.Row(*charts)

    def view(self):
        donations_view = self.donations_dashboard.view()
        donations_view[1].append(
            (
                'Subject Matter Expertise',
                pn.Column(
                    self.view_sme_tokens(),
                    self.view_sme(),
                ),
            )
        )
        donations_view[1].active = len(donations_view[1]) - 1

        view = pn.Row(
            pn.Column(
                self.tqf.param['boost_coefficient'],
                self.tqf.param['matching_pool'],
                self.tqf.param['matching_percentage_cap'],
            ),
            pn.Column(donations_view),
        )
        # Append a layout to the main area, to demonstrate the list-like API
        return view
