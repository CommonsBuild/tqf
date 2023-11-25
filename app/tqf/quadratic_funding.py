from io import BytesIO

import numpy as np
import pandas as pd
import panel as pn
import param as pm


class TunableQuadraticFunding(pm.Parameterized):

    donations = pm.Selector(doc='Donations Dataset')
    boost_factory = pm.Selector()
    boosts = pm.DataFrame(precedence=-1)
    boost_coefficient = pm.Number(1, bounds=(0, 10), step=0.1)
    matching_pool = pm.Integer(25000, bounds=(0, 250_000), step=5_000)
    matching_percentage_cap = pm.Magnitude(0.2, step=0.01)
    qf = pm.DataFrame(precedence=-1)
    boosted_donations = pm.DataFrame(precedence=-1)
    boosted_qf = pm.DataFrame(precedence=-1)
    results = pm.DataFrame(precedence=-1)
    mechanism = pm.Selector(
        objects=[
            'Direct Contributions',
            '1p1v',
            'Quadratic Funding',
            'Pairwise Penalty',
            'Cluster Mapping',
        ]
    )

    def _qf(self, donations_dataset, donation_column='amountUSD'):
        """Apply the quadratic algorithm."""
        qf = (
            donations_dataset.groupby(['Grant Name', 'grantAddress'])[donation_column]
            .apply(lambda x: np.square(np.sum(np.sqrt(x))))
            .sort_values(ascending=False)
        ).to_frame(name='quadratic_funding')

        # Calculate the relative distribution
        qf['distribution'] = qf / qf.sum()

        # Apply the Matching Percentage Cap
        qf['capped_distribution'] = qf['distribution'].where(
            qf['distribution'] < self.matching_percentage_cap,
            self.matching_percentage_cap,
        )

        # Applying a Cap to the Distribution

        # Identify Mask of Distributions Less than the Cap
        mask = qf['capped_distribution'] < self.matching_percentage_cap

        # Scale low distributions by 1 - sum of high distributions
        qf.loc[mask, 'capped_distribution'] *= (
            1 - qf['capped_distribution'][~mask].sum()
        ) / qf['capped_distribution'][mask].sum()

        # Cap the high distributions
        qf['capped_distribution'] = qf['capped_distribution'].where(
            qf['capped_distribution'] < self.matching_percentage_cap,
            self.matching_percentage_cap,
        )

        # Apply the Matching Pool
        qf['matching'] = qf['capped_distribution'] * self.matching_pool

        return qf

    @pm.depends(
        'donations.param',
        'matching_pool',
        'matching_percentage_cap',
        watch=True,
        on_init=True,
    )
    def update_qf(self):
        self.qf = self._qf(self.donations.dataset)

    def view_qf_bar(self):
        return self.qf['quadratic_funding'].hvplot.bar(
            title='Quadratic Funding', shared_axes=False
        )

    def view_qf_distribution_bar(self):
        return self.qf['distribution'].hvplot.bar(
            title='Quadratic Funding Distribution', shared_axes=False
        )

    def view_qf_matching_bar(self):
        return self.qf['matching'].hvplot.bar(
            title='Quadratic Funding Distribution', shared_axes=False
        )

    @pm.depends('boost_factory.param', watch=True, on_init=True)
    def update_boosts(self):
        self.boosts = self.boost_factory.collect_boosts()

    @pm.depends(
        'boosts', 'boost_coefficient', 'donations.dataset', watch=True, on_init=True
    )
    def update_boosted_donations(self):
        boosted_donations = self.boosts.merge(
            self.donations.dataset,
            left_on='address',
            right_on='voter',
            how='right',
        ).fillna(0)
        boosted_donations['coefficient'] = (
            1 + self.boost_coefficient * boosted_donations['Total_Boost']
        )
        boosted_donations['Boosted Amount'] = (
            boosted_donations['coefficient'] * boosted_donations['amountUSD']
        )
        self.boosted_donations = boosted_donations

    @pm.depends(
        'boosted_donations',
        'matching_pool',
        'matching_percentage_cap',
        watch=True,
        on_init=True,
    )
    def update_boosted_qf(self):
        boosted_qf = self._qf(self.boosted_donations, donation_column='Boosted Amount')
        self.boosted_qf = boosted_qf

    def donation_profile_clustermatch(self, donation_df, donation_column='amountUSD'):
        donation_df = self.donations.dataset.pivot_table(
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

    @pm.depends('qf', 'boosted_qf', watch=True, on_init=True)
    def update_results(self):
        # print('HEEERE')
        results = pd.merge(
            self.qf,
            self.boosted_qf,
            on=['Grant Name', 'grantAddress'],
            suffixes=('', '_boosted'),
        )
        # print(results)
        results['Boost Percentage Change'] = 100 * (
            (results['matching_boosted'] - results['matching']) / results['matching']
        )
        results['ClusterMatch'] = self.donation_profile_clustermatch(
            self.donations.dataset
        )

        results['Cluster Match Percentage Change'] = 100 * (
            (results['ClusterMatch'] - results['matching']) / results['matching']
        )
        results['ClusterMatch Boosted'] = self.donation_profile_clustermatch(
            self.boosted_donations, donation_column='amountUSD'
        )
        results['Cluster Match Boosted Percentage Change'] = 100 * (
            (results['ClusterMatch Boosted'] - results['matching'])
            / results['matching']
        )

        self.results = results[
            [
                'matching',
                'matching_boosted',
                'Boost Percentage Change',
                'ClusterMatch',
                'Cluster Match Percentage Change',
                'ClusterMatch Boosted',
                'Cluster Match Boosted Percentage Change',
            ]
        ].reset_index()

    def get_results_csv(self):
        output = BytesIO()
        self.results.reset_index().to_csv(output, index=False)
        output.seek(0)
        return output

    def get_boosted_donations_csv(self):
        output = BytesIO()
        boosted_donations = self.boosted_donations
        boosted_donations[self.donations.dataset.columns].reset_index().to_csv(
            output, index=False
        )
        output.seek(0)
        return output

    def view_results(self):
        return self.results

    def view(self):
        results_download = pn.widgets.FileDownload(
            callback=self.get_results_csv,
            filename='results.csv',
            button_type='primary',
        )
        boosted_donations_download = pn.widgets.FileDownload(
            callback=self.get_boosted_donations_csv,
            filename='boosted_donations.csv',
            button_type='primary',
        )
        return pn.Column(
            self, self.view_results, results_download, boosted_donations_download
        )
