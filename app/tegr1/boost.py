import pandas as pd
import panel as pn
import param as pm
from bokeh.models import NumeralTickFormatter


class Boost(pm.Parameterized):
    token_logy = pm.Boolean(
        False,
        doc='This parameter changes the token distribution view. It does not effect functionality.',
    )
    input = pm.Selector()
    signal = pm.Series(
        precedence=-1,
        doc='The input signal to the boost. This is generally a token distribution.',
    )
    distribution = pm.Series(
        constant=True, precedence=-1, doc='The resulting distribution.'
    )
    transformation = pm.Selector(
        default='Sigmoid',
        objects=['Threshold', 'MinMaxScale', 'NormalScale', 'Sigmoid'],
    )
    boost_factor = pm.Number(1, bounds=(0.1, 10), step=0.1)
    threshold = pm.Integer(default=100, precedence=-1, bounds=(0, 10_000), step=1)
    k = pm.Number(
        default=10,
        precedence=-1,
        bounds=(1, 20),
        doc='Steepness of the sigmoid curve',
        label='Steepness',
    )
    b = pm.Number(
        default=-0.2,
        precedence=-1,
        bounds=(-0.5, 0.5),
        doc='Shift of the sigmoid curve',
        label='Shift',
        step=0.01,
    )

    def __init__(self, **params):
        super().__init__(**params)
        self.input_signal()
        self.show_transformation_params()

    @pm.depends('input', watch=True)
    def input_signal(self):
        self.signal = self.input.dataset['balance']
        # Clip and set to bounds according to signal values
        # lower, upper = (1, min(self.param.threshold.bounds[1], int(self.signal.max())))
        # self.threshold = min(max(self.threshold, lower), upper)
        # self.param.threshold.softbounds = (lower, upper)

    @pm.depends(
        'signal', 'token_logy', 'threshold', 'k', 'b', 'boost_factor', watch=True
    )
    def update_distribution(self):
        signal = self.signal
        threshold = self.threshold

        with pm.edit_constant(self):
            with pm.parameterized.batch_call_watchers(self):
                if self.transformation == 'Threshold':
                    self.distribution = self._threshold(signal, threshold)
                elif self.transformation == 'Sigmoid':
                    self.distribution = self._sigmoid_scale(signal, k=self.k, b=self.b)
                elif self.transformation == 'MinMaxScale':
                    self.distribution = self._min_max_scale(signal)
                elif self.transformation == 'NormalScale':
                    self.distribution = self._normal_scale(signal)
                else:
                    raise (Exception(f'Unkown Transformation: {self.transformation}'))
                self.distribution = self.boost_factor * self.distribution

    @pm.depends('transformation', watch=True)
    def show_transformation_params(self):
        """
        This function controls which parameters are visible depending on which transformer is selected.
        """
        with pm.parameterized.batch_call_watchers(self):
            # Set all function parameters to not visible
            self.param['threshold'].precedence = -1
            self.param['k'].precedence = -1
            self.param['b'].precedence = -1

            if self.transformation == 'Threshold':
                self.param['threshold'].precedence = 1

            if self.transformation == 'Sigmoid':
                self.param['k'].precedence = 1
                self.param['b'].precedence = 1

            if self.transformation == 'MinMaxScale':
                pass

        self.update_distribution()

    @staticmethod
    def _sigmoid(x, A=1, k=1, b=0):
        """
        Parameters
        ----------
        x : The input value(s) for which the sigmoid function should be computed.
        A : The maximum value of the sigmoid curve.
        k : The steepness of the sigmoid curve.
        b : The x-axis shift of the sigmoid curve.
        """
        return A / (1 + np.exp(-k * (x + b)))

    @staticmethod
    def _min_max_scale(signal):
        return pd.Series((signal - signal.min()) / (signal.max() - signal.min()))

    @staticmethod
    def _threshold(signal, t):
        return (signal >= t).astype(int)

    @staticmethod
    def _normal_scale(signal):
        return ((signal - signal.mean()) / signal.std() + 0.5).clip(lower=0, upper=1)

    def _sigmoid_scale(self, signal, **params):
        return self._sigmoid(self._normal_scale(signal), **params)

    def view_distribution(self):
        distribution_view = (
            self.distribution.sort_values(ascending=False)
            .reset_index(drop=True)
            .hvplot.step(ylim=(-0.01, self.boost_factor + 0.01), title='Boost Factor')
            .opts(shared_axes=False, yformatter=NumeralTickFormatter(format='0.00'))
        )
        return distribution_view

    def view_signal(self):
        signal_view = (
            self.signal.sort_values(ascending=False)
            .reset_index(drop=True)
            .hvplot.step(logy=self.token_logy, title='Token Balance')
            .opts(shared_axes=False, yformatter=NumeralTickFormatter(format='0.a'))
        )
        return signal_view

    def view_explainer(self):
        if self.transformation == 'Threshold':
            return pn.pane.Markdown(f'The threshold is set to {self.threshold}.')

        if self.transformation == 'Sigmoid':
            explanation = f"""
                Sigmoid curve with steepness {self.k:.2f} and shift {self.b:.2f}.
                The sigmoid function is defined as:  
                $$ f(x) = \\frac{{1}}{{1 + e^{{-{self.k:.2f} \(x + {self.b:.2f}\)}}}} $$
                """

            return pn.pane.Markdown(explanation)

        if self.transformation == 'MinMaxScale':
            explanation = """
                Scale the distribution by min and max values.
                $$ f(x) = \\frac{{x - min(x)}}{{max(x) - min(x)}} $$
                """
            return pn.pane.Markdown(explanation)

        if self.transformation == 'NormalScale':
            explanation = """
                Scale the distribution by mean and standard deviation.'
                $$ f(x) = \\frac{{x - mean(x)}}{{std(x)}} + 0.5 $$
                """
            return pn.pane.Markdown(explanation)

    def view(self):
        return pn.Row(
            pn.Column(self, self.view_explainer),
            pn.Column(self.view_signal, self.view_distribution),
        )
