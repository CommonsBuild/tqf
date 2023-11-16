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
        objects=[
            'Threshold',
            'MinMaxScale',
            'LogMinMaxScale',
            'NormalScale',
            'LogNormalScale',
            'Sigmoid',
        ],
    )
    boost_factor = pm.Number(1, bounds=(0.1, 10), step=0.1)
    threshold = pm.Integer(default=100, precedence=-1, bounds=(0, 10_000), step=1)
    k = pm.Number(
        default=5,
        precedence=-1,
        bounds=(1, 20),
        doc='Steepness of the sigmoid curve',
        label='Steepness',
    )
    b = pm.Number(
        default=-0.2,
        precedence=-1,
        bounds=(-1, 1),
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
                elif self.transformation == 'LogNormalScale':
                    self.distribution = self._log_normal_scale(signal)
                elif self.transformation == 'LogMinMaxScale':
                    self.distribution = self._log_minmax_scale(signal)
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

    def _log_normal_scale(self, signal):
        return self._normal_scale(np.log(signal))

    def _log_minmax_scale(self, signal):
        return self._min_max_scale(np.log(signal))

    def _sigmoid_scale(self, signal, **params):
        return self._min_max_scale(
            self._sigmoid(self._log_minmax_scale(signal), **params)
        )

    @pm.depends('distribution')
    def view_distribution(self):
        distribution_view = (
            self.distribution.sort_values(ascending=False)
            .reset_index(drop=True)
            .hvplot.step(
                ylim=(-0.01, self.boost_factor + 0.01),
                title='Boost Factor',
                ylabel='boost',
            )
            .opts(
                shared_axes=False,
                yformatter=NumeralTickFormatter(format='0.00'),
                width=650,
                height=320,
            )
        )
        return distribution_view

    @pm.depends('signal', 'token_logy')
    def view_signal(self):
        signal_view = self.input.view_distribution().opts(
            shared_axes=False,
            yformatter=NumeralTickFormatter(format='0.a'),
            width=650,
            height=320,
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

    def view_boost(self):
        # Convert signal and distribution to DataFrames
        signal_df = self.signal.sort_values(ascending=False).reset_index(name='signal')
        distribution_df = self.distribution.sort_values(ascending=False).reset_index(
            name='distribution'
        )

        # Merge the two DataFrames on the index
        merged_df = pd.merge(signal_df, distribution_df, on='index')

        # If token_logy is True, transform the signal to a logarithmic scale
        if self.token_logy:
            # Replace 0 and negative values with a small positive value
            merged_df['signal'] = merged_df['signal'].replace(0, 1e-10)
            merged_df['signal'] = np.where(
                merged_df['signal'] <= 0, 1e-10, merged_df['signal']
            )

            # Apply the logarithmic transformation
            merged_df['signal'] = np.log1p(merged_df['signal'])

            # Scale the signal values
            min_val = self.signal.min()
            max_val = self.signal.max()
            merged_df['signal'] = (merged_df['signal'] - merged_df['signal'].min()) / (
                merged_df['signal'].max() - merged_df['signal'].min()
            ) * (max_val - min_val) + min_val

        # Create an Area plot for the signal and distribution
        signal_view = (
            hv.Area(merged_df, 'index', ['signal', 'distribution'])
            .redim.range(signal=(0.01, None))
            .opts(
                yformatter=NumeralTickFormatter(format='0.a'),
                width=650,
                height=320,
                logy=True,
            )
        )

        # Use datashade to shade the area plot based on the distribution values
        shaded_signal = datashade(signal_view, aggregator=ds.mean('distribution'))

        # Apply the desired options to the plot
        shaded_signal = shaded_signal.opts(
            title='Token Balance Boost Factor',
            shared_axes=False,
            yformatter=NumeralTickFormatter(format='0.a'),
            width=650,
            height=320,
            labelled=[],
        )

        # If token_logy is True, adjust the y-axis ticks to display the original values
        if self.token_logy:
            merged_df['signal'] = merged_df['signal'].clip(lower=1e-10)
            min_val = max(1e-10, self.signal.min())
            max_val = self.signal.max()
            ticks = [
                10**i
                for i in range(int(np.log10(min_val)), int(np.log10(max_val)) + 1)
            ]
            yticks = [(tick, str(tick)) for tick in ticks]
            shaded_signal = shaded_signal.options(yticks=yticks, logy=True)
        return shaded_signal

    @pm.depends('input', 'distribution')
    def output(self):
        df = self.input.dataset[['address', 'balance']].copy(deep=True)
        df['Boost'] = self.distribution
        df = df.sort_values(['Boost', 'balance'], ascending=False)
        return df

    @pm.depends('signal', 'distribution')
    def view_panel(self):
        return pn.Row(
            pn.Column(self, self.view_explainer),
            pn.Column(
                self.view_signal,
                self.view_distribution,
                # self.view_boost,
            ),
        )

    def view_output(self):
        return pn.Row(
            self.output,
        )

    def view(self):
        return self.view_panel() + self.view_output()
