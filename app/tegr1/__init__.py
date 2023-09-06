import logging
import threading

import hvplot.pandas
import numpy as np
import pandas as pd
import panel as pn
import param as pm
from icecream import ic
from IPython import start_ipython

pn.extension('mathjax')

# Use the following to initialize ipython when serving the app:
# def start_ipython_in_thread(namespace):
#     start_ipython(argv=[], user_ns=namespace)
#
#
# # Pass the main thread's namespace to the IPython instance
# ipython_thread = threading.Thread(target=start_ipython_in_thread, args=(globals(),))
# ipython_thread.start()


def exception_handler(ex):
    ic('🔥🔥🔥')
    logging.error('Error', exc_info=ex)
    pn.state.notifications.send(
        'Error: %s' % ex, duration=int(10e3), background='black'
    )


pn.extension(exception_handler=exception_handler, notifications=True)
pn.extension('tabulator')
pn.state.notifications.position = 'top-right'


class Dataset(pm.Parameterized):
    file = pm.FileSelector(
        default=None,
        path='./app/input/*.csv',
    )
    page_size = pm.Integer(default=20, precedence=-1)
    dataset = pm.DataFrame(
        default=None,
    )

    def __init__(self, **params):
        super().__init__(**params)
        self.load_file()

    @pm.depends('file', watch=True)
    def load_file(self):
        self.dataset = pd.read_csv(self.file)

    @pm.depends('dataset', 'page_size')
    def view(self):
        view = pn.panel(
            pn.Param(
                self.param,
                sort=False,
                widgets={
                    'dataset': {
                        'widget_type': pn.widgets.Tabulator,
                        'layout': 'fit_columns',
                        'page_size': self.page_size,
                        'pagination': 'remote',
                        'header_filters': True,
                    },
                },
            )
        )
        return view


class Donations(Dataset):
    file = pm.FileSelector(
        default='./app/input/vote_coefficients_input.csv',
        path='./app/input/*.csv',
    )
    dataset = pm.DataFrame(
        default=None, columns={'voter', 'amountUSD'}, label='Donations Dataset'
    )


donations = Donations()
import holoviews as hv


class DonationsDashboard(pm.Parameterized):
    donations = pm.Selector(default=donations, objects=[donations], precedence=-1)

    def donor_view(self):
        df = self.donations.dataset
        donor_vote_counts = (
            df.groupby('voter').count()['id'].to_frame(name='number_of_donations')
        )
        histogram = donor_vote_counts.hvplot.hist(
            ylabel='Donor Count',
            xlabel='Number of Projects Donated To',
            title='Number of Donations per Donor Histogram',
            height=320,
        )
        table = (
            donor_vote_counts.groupby('number_of_donations')
            .size()
            .reset_index(name='unique donor count')
            .sort_values('number_of_donations')
            .hvplot.table(height=320)
        )
        return pn.Row(histogram, table)

    def sankey_view(self):
        df = self.donations.dataset
        sankey = hv.Sankey(df[['voter', 'projectId', 'amountUSD']])
        return sankey

    def view(self):
        return pn.Column(
            self,
            pn.Tabs(
                ('Donor Donation Counts', self.donor_view),
                ('Sankey', self.sankey_view),
                active=0,
            ),
        )


donations_dashboard = DonationsDashboard()


class TokenDistribution(Dataset):
    file = pm.FileSelector(
        default=None,
        path='./app/input/*.csv',
    )
    dataset = pm.DataFrame(
        default=None, columns={'address', 'balance'}, label='Token Dataset'
    )


class TEC(TokenDistribution):
    file = pm.FileSelector(
        default='./app/input/tec_holders.csv',
        path='./app/input/*.csv',
    )


tec_distribution = TEC()


class TEA(TokenDistribution):
    file = pm.FileSelector(
        default='./app/input/tea_holders_dune.csv',
        path='./app/input/*.csv',
    )

    @pm.depends('file', watch=True)
    def load_file(self):
        """
        TEA Data might have 'wallet' field. If it does we remap it to 'address'.
        """
        df = pd.read_csv(self.file)
        df.rename({'wallet': 'address'}, axis=1, inplace=True)
        self.dataset = df


tea_distribution = TEA()

from bokeh.models import NumeralTickFormatter


class Boost(pm.Parameterized):
    token_logy = pm.Boolean(
        False,
        doc='This parameter changes the token distribution view. It does not effect functionality.',
    )
    input = pm.Selector(
        default=tec_distribution,
        objects=[tec_distribution, tea_distribution],
    )
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
            explanation = f"""
                Scale the distribution by min and max values.
                $$ f(x) = \\frac{{x - min(x)}}{{max(x) - min(x)}} $$
                """
            return pn.pane.Markdown(explanation)

        if self.transformation == 'NormalScale':
            explanation = f"""
                Scale the distribution by mean and standard deviation.'
                $$ f(x) = \\frac{{x - mean(x)}}{{std(x)}} + 0.5 $$
                """
            return pn.pane.Markdown(explanation)

    def view(self):
        return pn.Row(
            pn.Column(self, self.view_explainer),
            pn.Column(self.view_signal, self.view_distribution),
        )


tegr1_tec_boost = Boost(
    signal=tec_distribution.dataset['balance'],
    transformation='Threshold',
    threshold=10,
    input=tec_distribution,
    token_logy=True,
)

tegr1_tea_boost = Boost(
    signal=tec_distribution.dataset['balance'],
    transformation='Threshold',
    threshold=1,
    input=tea_distribution,
)


class BoostFactory(pm.Parameterized):
    template = pm.Selector(
        default=tegr1_tec_boost,
        objects=[tegr1_tec_boost, tegr1_tea_boost],
    )
    boosts = pm.List(default=[], class_=Boost, precedence=-1)
    new_boost = pm.Action(lambda self: self._new_boost())
    remove_boost = pm.Action(lambda self: self._remove_boost())

    def _new_boost(self):
        self.boosts.append(Boost(**self.template.param.values()))
        self.param.trigger('boosts')

    def _remove_boost(self):
        if len(self.boosts):
            self.boosts.pop()
            self.param.trigger('boosts')

    @pm.depends('boosts', watch=True)
    def boosts_view(self):
        return pn.Column(*[boost.view for boost in self.boosts])

    @pm.depends('boosts')
    def view(self):
        return pn.Row(self, self.boosts_view)


boost_factory = BoostFactory()


class QuadraticFunding(pm.Parameterized):

    donations = pm.Selector(
        default=donations,
        objects=[donations],
        precedence=-1,
    )

    def view(self):
        return pn.Column(self, self.donations.view)


qf = QuadraticFunding()

app = pn.Tabs(
    ('Donations', pn.Column(donations.view(), donations_dashboard.view())),
    ('Quadradic Funding', qf.view()),
    ('Token Distribution', tec_distribution.view()),
    ('TEA Token Distribution', tea_distribution.view()),
    ('SME Signal Boost', tegr1_tec_boost.view()),
    ('Boost Factory', boost_factory.view()),
    active=0,
)


if __name__ == '__main__':
    print(donations)
