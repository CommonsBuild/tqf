import numpy as np
import pandas as pd
import panel as pn
import param as pm


class TunableQuadraticFunding(pm.Parameterized):

    donations = pm.Selector()
    boost_factory = pm.Selector()
    boosts = pm.DataFrame(precedence=-1)
    boost_coefficient = pm.Number(1, bounds=(0, 10), step=0.1)
    matching_pool = pm.Integer(25000, bounds=(0, 250_000), step=5_000)
    matching_percentage_cap = pm.Magnitude(0.2, step=0.01)
    qf = pm.DataFrame(precedence=-1)
    boosted_donations = pm.DataFrame(precedence=-1)
    boosted_qf = pm.DataFrame(precedence=-1)
    results = pm.DataFrame(precedence=1)

    def _qf(self, donations_dataset, donation_column='amountUSD'):
        """Apply the quadratic algorithm."""
        qf = (
            donations_dataset.groupby('applicationId')[donation_column]
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
        mask = qf['capped_distribution'] < self.matching_percentage_cap
        qf.loc[mask, 'capped_distribution'] *= (
            1 - qf['capped_distribution'][~mask].sum()
        ) / qf['capped_distribution'][mask].sum()
        qf['capped_distribution'] = qf['capped_distribution'].where(
            qf['capped_distribution'] < self.matching_percentage_cap,
            self.matching_percentage_cap,
        )

        # Apply the Matching Pool
        qf['matching'] = qf['capped_distribution'] * self.matching_pool
        return qf

    @pm.depends(
        'donations',
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
        boosted_donations['Boosted Amount'] = (
            1 + self.boost_coefficient * boosted_donations['Total_Boost']
        ) * boosted_donations['amountUSD']
        self.boosted_donations = boosted_donations

    @pm.depends(
        'boosted_donations',
        'matching_pool',
        'matching_percentage_cap',
        watch=True,
    )
    def update_boosted_qf(self):
        boosted_qf = self._qf(self.boosted_donations, donation_column='Boosted Amount')
        self.boosted_qf = boosted_qf

    @pm.depends('qf', 'boosted_qf', watch=True)
    def update_results(self):
        results = pd.merge(
            self.qf,
            self.boosted_qf,
            on='applicationId',
            suffixes=('', '_boosted'),
        )
        results['Percentage Boost'] = 100 * (
            (results['matching_boosted'] - results['matching']) / results['matching']
        )
        self.results = results

    def view(self):
        return pn.Column(self)
