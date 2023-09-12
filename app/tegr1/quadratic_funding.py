import numpy as np
import panel as pn
import param as pm


class QuadraticFunding(pm.Parameterized):

    donations = pm.Selector(
        precedence=-1,
    )

    def qf(self):
        """Apply the quadratic algorithm."""
        qf = self.donations.dataset.groupby('applicationId')['amountUSD'].apply(
            lambda x: np.square(np.sum(np.sqrt(x)))
        )
        return qf

    def qf_view(self):
        return self.qf().hvplot.bar(title='Quadratic Funding')

    def view(self):
        return pn.Column(self, self.donations.view, self.qf_view)
