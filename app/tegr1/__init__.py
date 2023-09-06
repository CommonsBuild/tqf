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
    logy = pm.Boolean(False)
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
    threshold = pm.Number(default=100, precedence=-1, bounds=(0, 10_000), step=1)
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

    @pm.depends('signal', 'logy', 'threshold', 'k', 'b', watch=True)
    def update_distribution(self):
        # if self.logy:
        #     signal = np.log(self.signal + 1)
        #     threshold = np.log(self.threshold)
        # else:
        #     signal = self.signal
        #     threshold = self.threshold
        signal = self.signal
        threshold = self.threshold

        with pm.edit_constant(self):
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
            .hvplot.step(ylim=(-0.01, 1.01), title='Boost Factor')
            .opts(shared_axes=False, yformatter=NumeralTickFormatter(format='0.00'))
        )
        return distribution_view

    def view_signal(self):
        signal_view = (
            self.signal.sort_values(ascending=False)
            .reset_index(drop=True)
            .hvplot.step(logy=self.logy, title='Token Balance')
            .opts(shared_axes=False, yformatter=NumeralTickFormatter(format='0.a'))
        )
        return signal_view

    def view_explainer(self):
        if self.transformation == 'Threshold':
            return pn.pane.Markdown(f'The threshold is set to {self.threshold}.')

        if self.transformation == 'Sigmoid':
            explanation = f"""
                Sigmoid curve with steepness {self.k:.2f} and shift {self.b:.2f}.
                $$ f(x) = \\frac{{1}}{{1 + e^{{-k(x + b)}}}} $$
                """

            return pn.pane.Markdown(explanation)

        if self.transformation == 'MinMaxScale':
            return pn.pane.Markdown(f'Scale the distribution by min and max values.')

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

app = pn.Tabs(
    ('Donations', donations.view()),
    ('Token Distribution', tec_distribution.view()),
    ('TEA Token Distribution', tea_distribution.view()),
    ('SME Signal Boost', tegr1_tec_boost.view()),
    ('Boost Factory', boost_factory.view()),
    active=3,
)


if __name__ == '__main__':
    print(donations)
