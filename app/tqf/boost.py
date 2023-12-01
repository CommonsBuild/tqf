import datashader as ds
import holoviews as hv
import numpy as np
import pandas as pd
import panel as pn
import param as pm
from bokeh.models import BasicTicker, LogTicker, NumeralTickFormatter
from holoviews.operation.datashader import datashade, dynspread, shade

pn.extension('mathjax')


class Boost(pm.Parameterized):
    distribution = pm.Selector(
        doc='The input token distribution.',
        instantiate=True,
        per_instance=True,
    )
    threshold = pm.Integer(
        default=1,
        precedence=1,
        bounds=(1, 1000),
        step=1,
        doc='Minimum number of tokens required to qualify for this boost.',
        instantiate=True,
        per_instance=True,
    )
    transformation = pm.Selector(
        default='Sigmoid',
        objects=[
            'Threshold',
            'Linear',
            'LogLinear',
            'Normal',
            'LogNormal',
            'Sigmoid',
        ],
        doc='Select the transformation to apply to the distribution.',
    )
    k = pm.Number(
        default=-5,
        precedence=-1,
        bounds=(-10, -1),
        doc='Steepness of the sigmoid curve',
        label='k: Sigmoid Steepness',
        step=0.001,
    )
    b = pm.Number(
        default=1,
        step=1,
        precedence=-1,
        bounds=(-10, 10),
        doc='Shift of the sigmoid curve',
        label='b: Sigmoid Shift',
    )
    max_boost = pm.Number(
        2, bounds=(1, 10), step=0.1, doc='Scaling factor for this boost.'
    )
    boost = pm.Series(precedence=-1, doc='The resulting boost coefficient.')

    @staticmethod
    def _threshold(x, t):
        """
        Threshold.
        Parameters
        ----------
        x : ...
        """

        return (x >= t).astype(int)

    @staticmethod
    def _linear_scale(x, t):
        """
        Min-max scale.
        Parameters
        ----------
        x : ...
        """

        print('Linear SCALE')
        linear = pd.Series((x - t + 1) / (x.max() - t + 1))

        print(linear.min())
        print(linear.max())
        print(linear.mean())

        return linear

    @staticmethod
    def _normal_scale(x, t):
        """
        Normal scale.
        Parameters
        ----------
        x : ...
        """
        mu = x[x >= t].mean()
        sigma = x[x >= t].std()
        normal = pd.Series(x - mu) / sigma
        return normal.clip(lower=0, upper=1)

    @staticmethod
    def _sigmoid(x, k, b):
        return 1 / (1 + np.exp(-np.exp(k) * (x + b)))

    def threshold_boost(self):
        x = self.distribution.dataset['balance']
        t = self.threshold
        return self._threshold(x, t)

    def linear_boost(self):
        x = self.distribution.dataset['balance']
        t = self.threshold
        return self._linear_scale(x, t)

    def log_linear_boost(self):
        x = self.distribution.dataset['balance']
        t = self.threshold
        return self._linear_scale(np.log(x + 1), np.log(t + 1))

    def normal_boost(self):
        x = self.distribution.dataset['balance']
        t = self.threshold
        return self._normal_scale(x, t)

    def log_normal_boost(self):
        x = self.distribution.dataset['balance']
        t = self.threshold
        return self._normal_scale(np.log(x + 1), np.log(t + 1))

    def sigmoid_boost(self):
        x = self.distribution.dataset['balance']
        k = self.k
        b = self.b
        t = self.threshold
        return self._sigmoid(x, k=k, b=b)

    def __init__(self, **params):
        threshold = params.get('threshold', 1)
        threshold = min(threshold, self.param['threshold'].bounds[1])
        params['threshold'] = threshold
        super(Boost, self).__init__(**params)
        self.set_bounds()

    @pm.depends('distribution', watch=True, on_init=True)
    def set_bounds(self):
        balances = self.distribution.dataset['balance']
        max_balance = min(int(balances.max()), 1000)
        self.threshold = max(1, min(self.threshold, max_balance))
        self.param['threshold'].bounds = (
            1,
            max_balance,
        )
        self.b = max(-max_balance, min(self.b, max_balance))
        self.param['b'].bounds = (
            -max_balance,
            max_balance,
        )
        self.param.trigger('threshold', 'b')

    @pm.depends('transformation', watch=True, on_init=True)
    def set_transformation_params_visibility(self):
        """
        This function controls which parameters are visible depending on which transformer is selected.
        """
        with pm.parameterized.batch_call_watchers(self):
            self.param['k'].precedence = -1
            self.param['b'].precedence = -1
            if self.transformation in ['Sigmoid']:
                self.param['k'].precedence = 1
                self.param['b'].precedence = 1

    @pm.depends(
        'distribution',
        'max_boost',
        'transformation',
        'threshold',
        'k',
        'b',
        watch=True,
        on_init=True,
    )
    def update_boost(self):
        # This keeps the update efficient, ensuring that events are all triggered at once.
        with pm.parameterized.batch_call_watchers(self):

            transformations = {
                'Threshold': self.threshold_boost,
                'Linear': self.linear_boost,
                'LogLinear': self.log_linear_boost,
                'Normal': self.normal_boost,
                'LogNormal': self.log_normal_boost,
                'Sigmoid': self.sigmoid_boost,
            }

            # Loads the selected transformation.
            transformation_function = transformations[self.transformation]

            self.boost = (
                1 + (self.max_boost - 1) * transformation_function()
            ) * self.threshold_boost()

            self.distribution.dataset['boost'] = self.boost

    @pm.depends('boost')
    def view_boost(self):
        boost_view = (
            self.boost.sort_values(ascending=False)
            .reset_index(drop=True)
            .hvplot.area(
                ylim=(0, self.max_boost),
                title='Boost',
                ylabel='boost',
                height=400,
                responsive=True,
                shared_axes=False,
                yformatter=NumeralTickFormatter(format='0.00'),
                xlabel='token_holders',
            )
        )
        return boost_view

    def view(self):
        return pn.Row(
            self,
            self.view_boost,
        )
