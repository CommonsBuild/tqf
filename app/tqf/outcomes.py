import hvplot.pandas
import numpy as np
import pandas as pd
import panel as pn
import param as pm


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
        boost_cols = [c for c in collected_boosts.columns if 'Boost_' in c]
        balance_cols = [c for c in collected_boosts.columns if 'balance' in c]
        cols_list = list(zip(['address'] * len(balance_cols), balance_cols, boost_cols))

        donations = self.donations_dashboard.donations.dataset

        charts = []

        contributors = self.donations_dashboard.contributor_dataset()

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

    def view(self):
        donations_view = self.donations_dashboard.view()
        donations_view[1].append(
            (
                'Subject Matter Expertise',
                pn.Column(self.boost_factory.view_outcomes(), self.view_sme()),
            )
        )
        donations_view[1].active = len(donations_view[1]) - 1

        view = pn.Row(
            pn.Column(
                self.tqf.param['boost_coefficient'],
                self.tqf.param['matching_pool'],
                self.tqf.param['matching_percentage_cap'],
            ),
            pn.Column(
                # self.boost_factory.view_outcomes(),
                # pd.DataFrame([-1, 0, 1]).hvplot.scatter(),
                # self.donations_dashboard.view().append(('Test', 'Test')),
                donations_view
            ),
        )
        # Append a layout to the main area, to demonstrate the list-like API
        return view
