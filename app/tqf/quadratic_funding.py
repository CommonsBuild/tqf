from io import BytesIO
from math import log

import numpy as np
import pandas as pd
import panel as pn
import param as pm
from bokeh.models import HoverTool
from bokeh.palettes import Greens

Greens = Greens[256][::-1]


class TunableQuadraticFunding(pm.Parameterized):

    donations_dashboard = pm.Selector(doc='Donations Dataset', precedence=-1)
    boost_factory = pm.Selector(precedence=-1)
    boosts = pm.DataFrame(precedence=-1)
    matching_pool = pm.Integer(25000, bounds=(0, 250_000), step=5_000)
    matching_percentage_cap = pm.Number(0.2, step=0.01, bounds=(0.01, 1))
    qf = pm.DataFrame(precedence=-1)
    boosted_donations = pm.DataFrame(precedence=-1)
    boosted_qf = pm.DataFrame(precedence=-1)
    results = pm.DataFrame(precedence=-1)
    mechanism = pm.Selector(
        default='Quadratic Funding',
        objects=[
            'Direct Donations',
            # '1p1v',
            'Quadratic Funding',
            # 'Pairwise Penalty',
            'Cluster Mapping',
        ],
    )
    project_stats = pm.DataFrame(precedence=-1)

    def _qf(
        self,
        donations_dataset,
        donation_column='amountUSD',
        mechanism='Quadratic Funding',
    ):
        # Group by 'Grant Name' and 'grantAddress', then apply calculations
        qf = donations_dataset.groupby(['Grant Name', 'grantAddress']).apply(
            lambda group: pd.Series(
                {
                    'Funding Mechanism': np.square(
                        np.sum(np.sqrt(group[donation_column]))
                    ),
                    'Direct Donations': group['amountUSD'].sum(),
                    'Boosted Donations': group[donation_column].sum(),
                }
            )
        )

        if mechanism == 'Direct Donations':
            qf['Funding Mechanism'] = 2 * qf['Boosted Donations']
        qf = qf.drop('Boosted Donations', axis=1)

        if mechanism == 'Cluster Mapping':
            qf['Funding Mechanism'] = self.donation_profile_clustermatch(
                donations_dataset,
                donation_column=donation_column,
            )

        # Sort values if needed
        qf = qf.sort_values(by='Funding Mechanism', ascending=False)

        # Calculate 'Matching Funding'
        qf['Matching Funding'] = qf['Funding Mechanism'] - qf['Direct Donations']

        # Calculate the stochastic vector funding distribution
        qf['Matching Distribution'] = (
            qf['Matching Funding'] / qf['Matching Funding'].sum()
        )

        # Applying a Cap to the Distribution
        qf['Matching Distribution'] = qf['Matching Distribution'].where(
            qf['Matching Distribution'] < self.matching_percentage_cap,
            self.matching_percentage_cap,
        )

        # Identify Mask of Distributions Less than the Cap
        mask = qf['Matching Distribution'] < self.matching_percentage_cap

        # Scale low distributions by 1 - sum of high distributions
        qf.loc[mask, 'Matching Distribution'] *= (
            1 - qf['Matching Distribution'][~mask].sum()
        ) / qf['Matching Distribution'][mask].sum()

        # Cap the high distributions
        qf['Matching Distribution'] = qf['Matching Distribution'].where(
            qf['Matching Distribution'] < self.matching_percentage_cap,
            self.matching_percentage_cap,
        )

        # Apply the Matching Pool
        qf['Matching Funds'] = qf['Matching Distribution'] * self.matching_pool

        # Apply the Matching Pool
        qf['Total Funding'] = qf['Matching Funds'] + qf['Direct Donations']

        return qf

    @pm.depends(
        'donations_dashboard.donations.param',
        'matching_pool',
        'matching_percentage_cap',
        'mechanism',
        watch=True,
        on_init=True,
    )
    def update_qf(self):
        self.qf = self._qf(
            self.donations_dashboard.donations.dataset, mechanism=self.mechanism
        )

    # def view_qf_bar(self):
    #     return self.results['quadratic_funding'].hvplot.bar(
    #         title='Quadratic Funding', shared_axes=False
    #     )
    #
    # def view_qf_distribution_bar(self):
    #     return self.results['distribution'].hvplot.bar(
    #         title='Quadratic Funding Distribution', shared_axes=False
    #     )

    def view_qf_matching_bar(self):
        def truncate_string(s, max_length=40):
            return s if len(s) <= max_length else s[: max_length - 3] + '...'

        results = self.results.sort_values('Matching Funds Boosted', ascending=False)
        results['Grant Name'] = results['Grant Name'].apply(truncate_string)
        results = results.sort_values('Matching Funds Boosted', ascending=False).rename(
            columns={
                'Matching Funds Boosted': 'matching_funds',
                'Grant Name': 'grant_name',
            }
        )

        hover = HoverTool(
            tooltips=[
                ('Grant Name', '@grant_name'),
                ('Matching Funds Boosted', '$@matching_funds{0,0}'),
            ]
        )

        chart = results.hvplot.bar(
            title='Matching Funds Boosted',
            y='matching_funds',
            x='grant_name',
            color='matching_funds',
            shared_axes=False,
            height=800,
            width=1600,
            cmap='RdYlGn',
            rot=90,
            ylabel='Matching Funds Boosted',
            xlabel='Grant Name',
            tools=[hover],
        )

        chart2 = results.hvplot.bar(
            x='grant_name',
            y='Matching Funds Not Boosted',
            color='gray',
            xlabel=None,
            ylabel=None,
            line_width=2,
            fill_alpha=0.2,
        )

        return chart * chart2

    @pm.depends('boost_factory.param', watch=True, on_init=True)
    def update_boosts(self):
        self.boosts = self.boost_factory.boost_outputs

    @pm.depends(
        'boosts',
        'donations_dashboard.donations.dataset',
        watch=True,
        on_init=True,
    )
    def update_boosted_donations(self):
        # Merge Boosts into Donations
        boosted_donations = self.donations_dashboard.donations.dataset.merge(
            self.boosts,
            how='left',
            left_on='voter',
            right_on='address',
        )
        if set(['projectId_x', 'projectId_y']).issubset(boosted_donations.columns):
            boosted_donations = boosted_donations.rename(
                {'projectId_x': 'projectId'}, axis=1
            ).drop('projectId_y', axis=1)

        # Non-boosted donations are initially set to 0
        boosted_donations = boosted_donations.fillna(0)

        # Set the Boost Coefficient
        boosted_donations['Boost Coefficient'] = 1 + boosted_donations['total_boost']

        # Set the Boosted Amount as a Boost Coefficient * Donation Amount
        boosted_donations['Boosted Amount'] = (
            boosted_donations['Boost Coefficient'] * boosted_donations['amountUSD']
        )

        # Set the Boosted Donations on the TQF Class Instance
        self.boosted_donations = boosted_donations

    @pm.depends(
        'boosted_donations',
        'matching_pool',
        'matching_percentage_cap',
        'mechanism',
        watch=True,
    )
    def update_boosted_qf(self):
        boosted_qf = self._qf(
            self.boosted_donations,
            donation_column='Boosted Amount',
            mechanism=self.mechanism,
        )
        self.boosted_qf = boosted_qf

    def donation_profile_clustermatch(self, donation_df, donation_column='amountUSD'):
        donation_df = donation_df.pivot_table(
            index='voter',
            columns=['Grant Name', 'grantAddress'],
            values=donation_column,
        ).fillna(0)
        # Convert donation dataframe to binary dataframe
        binary_df = (donation_df > 0).astype(int)

        # Create 'cluster' column representing the donation profile of each donor
        binary_df['cluster'] = binary_df.apply(
            lambda row: ''.join(row.astype(str)), axis=1
        )

        # Group by 'cluster' and sum donations from the same cluster
        cluster_sums = donation_df.groupby(binary_df['cluster']).sum()

        # Calculate the square root of each donation in the cluster
        cluster_sqrt = np.sqrt(cluster_sums)

        # Sum the square roots of all clusters grouped by project and square the sums
        funding = (cluster_sqrt.sum() ** 2).to_dict()

        return funding

    @pm.depends('boosted_donations', watch=True, on_init=True)
    def update_project_stats(self):
        projects = self.donations_dashboard.projects_table(
            donations_df=self.donations_dashboard.donations.dataset,
            donation_column='amountUSD',
        )
        boosted_projects = self.donations_dashboard.projects_table(
            donations_df=self.boosted_donations, donation_column='Boosted Amount'
        )
        sme_donations = self.boosted_donations[self.boosted_donations['address'] != 0]
        total_smes = sme_donations['address'].nunique()
        total_sme_donations = sme_donations['Boosted Amount'].sum()
        sme_stats = sme_donations.groupby('Grant Name').apply(
            lambda group: pd.Series(
                {
                    'Number of SMEs': group['address'].nunique(),
                    'Percentage of SMEs': group['address'].nunique() / total_smes,
                    'Total SME Donations': group['Boosted Amount'].sum(),
                    'Percent of Total SME Donations': group['Boosted Amount'].sum()
                    / total_sme_donations,
                    'Mean SME Donation': group['Boosted Amount'].mean(),
                    'Median SME Donation': group['Boosted Amount'].median(),
                    'Max SME Donations': group['Boosted Amount'].max(),
                    'Max SME Donor': group.loc[
                        group['Boosted Amount'].idxmax(), 'voter'
                    ],
                    'SMEs': [
                        a[:8] for a in sorted(group['address'].tolist(), reverse=True)
                    ],
                    'SME Donations': sorted(
                        group['Boosted Amount'].tolist(), reverse=True
                    ),
                }
            )
        )

        projects = projects.merge(
            boosted_projects,
            how='inner',
            left_on='Grant Name',
            right_on='Grant Name',
            suffixes=(' Not Boosted', ' Boosted'),
        ).merge(sme_stats, how='outer', left_on='Grant Name', right_index=True)
        self.project_stats = projects

    @pm.depends('donations.dataset')
    def projects_table(self, donations_df, donation_column='amountUSD'):

        total_donations = donations_df[donation_column].sum()
        total_donors = donations_df['voter'].nunique()

        # Calculate Data per Project
        projects = (
            donations_df.groupby('Grant Name')
            .apply(
                lambda group: pd.Series(
                    {
                        'Number of Donors': group['voter'].nunique(),
                        'Percentage of Donors': group['voter'].nunique() / total_donors,
                        'Total Donations': group[donation_column].sum(),
                        'Percent of Total Donations': group[donation_column].sum()
                        / total_donations,
                        'Mean Donation': group[donation_column].mean(),
                        'Median Donation': group[donation_column].median(),
                        'Max Donations': group[donation_column].max(),
                        'Max Donor': group.loc[
                            group[donation_column].idxmax(), 'voter'
                        ],
                        'Donations': sorted(
                            group[donation_column].tolist(), reverse=True
                        ),
                    }
                )
            )
            .reset_index()
        )

        return projects

    @pm.depends('qf', 'boosted_qf', watch=True)
    def update_results(self):
        if (
            self.project_stats is not None
            and self.qf is not None
            and self.boosted_qf is not None
        ):
            stats = self.project_stats.reset_index()
            qf_results = pd.merge(
                self.qf,
                self.boosted_qf,
                on=['Grant Name', 'grantAddress'],
                suffixes=(' Not Boosted', ' Boosted'),
            )
            results = stats.drop(
                [
                    'Max Donor Not Boosted',
                    'Donations Not Boosted',
                    'Max Donor Boosted',
                    'Donations Boosted',
                    'SMEs',
                    'SME Donations',
                    'Max SME Donor',
                ],
                axis=1,
            ).merge(
                qf_results,
                left_on='Grant Name',
                right_on='Grant Name',
            )
            results['Matching Funds Boost Percentage'] = 100 * (
                (
                    results['Matching Funds Boosted']
                    - results['Matching Funds Not Boosted']
                )
                / results['Matching Funds Not Boosted']
            ).round(4)
            results['Total Funding Boost Percentage'] = 100 * (
                (
                    results['Total Funding Boosted']
                    - results['Total Funding Not Boosted']
                )
                / results['Total Funding Not Boosted']
            ).round(4)

            self.results = results

    def get_results_csv(self):
        output = BytesIO()
        self.results.reset_index().to_csv(output, index=False)
        output.seek(0)
        return output

    def get_boosted_donations_csv(self):
        output = BytesIO()
        boosted_donations = self.boosted_donations
        boosted_donations[
            self.donations_dashboard.donations.dataset.columns
        ].reset_index().to_csv(output, index=False)
        output.seek(0)
        return output

    def view_results(self):
        results = (
            self.results.rename(
                {'Total Donations Not Boosted': 'Direct Donations'}, axis=1
            )[
                [
                    'Grant Name',
                    'Direct Donations',
                    'Matching Funds Not Boosted',
                    'Matching Funds Boosted',
                    'Total Funding Boosted',
                    'Matching Funds Boost Percentage',
                ]
            ]
            .sort_values('Matching Funds Boosted', ascending=False)
            .reset_index(drop=True)
        )
        numeric_columns = results.select_dtypes(include=['number']).columns
        results_view = pn.widgets.Tabulator(
            results,
            pagination=None,
        )

        def color_cell(cell_value, min_value, max_value):
            if (cell_value <= 0) or np.isnan(cell_value):
                style = 'background-color: white; color: black;'
                print(style)
                return style
            # Normalize the logarithmic value and map it to the reversed Greens palette
            normalized_value = (log(cell_value) - log(min_value)) / (
                log(max_value) - log(min_value)
            )
            color_index = int(normalized_value * (len(Greens) - 1))

            # Determine text color based on background darkness
            text_color = 'white' if color_index > len(Greens) // 2 else 'black'

            style = f'background-color: {Greens[color_index]}; color: {text_color};'
            print(style)
            return style

        def color_cell(cell_value):
            style = 'background-color: black; color: white;'
            print(style)
            return style

        def color_series(s):
            print('here is s:')
            print(s)
            s_log = s.apply(log)
            print('here is s_log:')
            print(s_log)
            s_norm = (s_log - s_log.min()) / (s_log.max() - s_log.min())
            print('here is s_norm:')
            print(s_norm)
            s_color_index = (s * (len(Greens) - 1)).astype(int)
            print('here is s_color_index:')
            print(s_color_index)
            s_text_color = np.where(s_color_index > len(Greens) // 2, 'white', 'black')
            print('here is s_text_color:')
            print(s_text_color)
            s_style = [
                f'background-color: {Greens[color_index]}; color: {text_color};'
                for color_index, text_color in zip(s_color_index, s_text_color)
            ]
            print('here is s_style:')
            print(s_style)
            return s_style

        for col in numeric_columns:
            # Set the minimum and maximum values for the colormap
            min_value = results[col].replace(0, np.nan).min().min()
            max_value = results[col].max().max()
            # results_view.style.map(
            #     lambda cell_value: color_cell(cell_value, min_value, max_value),
            #     subset=[col],
            # )
            # results_view.style.map(
            #     color_cell,
            #     subset=[col],
            # )
            results_view.style.apply(
                color_series,
                subset=[col],
            )
        results_view.style.apply(
            color_series,
            subset=[numeric_columns],
        )

        return results_view

    def view_results_bar(self):
        def truncate_string(s, max_length=30):
            return s if len(s) <= max_length else s[: max_length - 3] + '...'

        results = self.results.sort_values(
            'Matching Funds Boost Percentage', ascending=False
        )
        results['Grant Name'] = results['Grant Name'].apply(truncate_string)
        return results.hvplot.bar(
            title='Matching Funds Boost Percentage',
            x='Grant Name',
            y='Matching Funds Boost Percentage',
            c='Matching Funds Boost Percentage',
            cmap='RdYlGn',
            ylim=(-100, 100),
            colorbar=False,
            rot=90,
            height=400,
            width=1600,
            fontscale=1,
            grid=True,
        )

    def view(self):
        boosted_donations_download = pn.widgets.FileDownload(
            callback=self.get_boosted_donations_csv,
            filename='boosted_donations.csv',
            button_type='primary',
        )
        results_download = pn.widgets.FileDownload(
            callback=self.get_results_csv,
            filename='results.csv',
            button_type='primary',
        )
        return pn.Column(
            self, self.view_results, boosted_donations_download, results_download
        )
