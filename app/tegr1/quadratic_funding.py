import numpy as np
import panel as pn
import param as pm


class QuadraticFunding(pm.Parameterized):

    donations = pm.Selector(
        precedence=-1,
    )
    matching_pool = pm.Integer(25000, bounds=(0, 250_000), step=5_000)
    matching_percentage_cap = pm.Magnitude(0.2, step=0.01)
    qf = pm.DataFrame()

    def _qf(self, donations):
        """Apply the quadratic algorithm."""
        qf = (
            self.donations.dataset.groupby('applicationId')['amountUSD']
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
        self.qf = self._qf(self.donations)

    def qf_bar(self):
        return self.qf['quadratic_funding'].hvplot.bar(
            title='Quadratic Funding', shared_axes=False
        )

    def qf_distribution_bar(self):
        return self.qf['distribution'].hvplot.bar(
            title='Quadratic Funding Distribution', shared_axes=False
        )

    def qf_matching_bar(self):
        return self.qf['matching'].hvplot.bar(
            title='Quadratic Funding Distribution', shared_axes=False
        )

    def view(self):
        return pn.Column(
            self,
            pn.Row(
                # self.qf,
                # pn.Column(self.qf_bar, self.qf_distribution_bar, self.qf_matching_bar),
            ),
        )
