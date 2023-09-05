import logging

import hvplot.pandas
import numpy as np
import pandas as pd
import panel as pn
import param as pm
from icecream import ic


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


class Boost(pm.Parameterized):
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
    logy = pm.Boolean(False)
    transformation = pm.ObjectSelector(
        'Sigmoid', objects=['Threshold', 'Linear', 'Sigmoid']
    )
    threshold = pm.Number(100, precedence=-1, bounds=(0, 1000), step=1)
    sigmoid_frequency = pm.Number(1, precedence=-1, bounds=(0.1, 5))
    sigmoid_shift = pm.Number(0, precedence=-1, bounds=(-5, 5))

    def __init__(self, **params):
        super().__init__(**params)
        self.input_signal()
        self.show_transformation_params()

    @pm.depends('input', watch=True)
    def input_signal(self):
        self.signal = self.input.dataset['balance']

    @pm.depends(
        'signal', 'logy', 'threshold', 'sigmoid_frequency', 'sigmoid_shift', watch=True
    )
    def update_distribution(self):
        if self.logy:
            signal = np.log(self.signal + 1)
            threshold = np.log(self.threshold)
        else:
            signal = self.signal
            threshold = self.threshold

        with pm.edit_constant(self):
            if self.transformation == 'Threshold':
                self.distribution = self._threshold(signal, threshold)
            elif self.transformation == 'Sigmoid':
                self.distribution = self._sigmoid_scale(
                    signal, k=self.sigmoid_frequency, b=self.sigmoid_shift
                )
            elif self.transformation == 'Linear':
                self.distribution = self._min_max_scale(signal)
            else:
                raise (Exception(f'Unkown Transformation: {self.transformation}'))

    @pm.depends('transformation', watch=True)
    def show_transformation_params(self):

        with pm.parameterized.batch_call_watchers(self):
            self.param['threshold'].precedence = -1
            self.param['sigmoid_frequency'].precedence = -1
            self.param['sigmoid_shift'].precedence = -1

            if self.transformation == 'Threshold':
                self.param['threshold'].precedence = 1

            if self.transformation == 'Sigmoid':
                self.param['sigmoid_frequency'].precedence = 1
                self.param['sigmoid_shift'].precedence = 1

        self.update_distribution()

    @staticmethod
    def _sigmoid(x, A=1, k=1, b=0):
        return A / (1 + np.exp(-k * (x - b)))

    @staticmethod
    def _min_max_scale(signal):
        return pd.Series((signal - signal.min()) / (signal.max() - signal.min()))

    @staticmethod
    def _threshold(signal, t):
        return (signal >= t).astype(int)

    @staticmethod
    def _mean_std_scale(signal):
        return (signal - signal.mean()) / signal.std()

    def _sigmoid_scale(self, signal, **params):
        return self._min_max_scale(
            self._sigmoid(self._mean_std_scale(signal), **params)
        )

    def view_distribution(self):
        return (
            self.distribution.sort_values(ascending=False)
            .reset_index(drop=True)
            .hvplot.step()
        )

    def view(self):
        return pn.Row(self, self.view_distribution)


# boost = Boost(input=tec_distribution)
boost = Boost(signal=tec_distribution.dataset['balance'])


class BoostFactory(pm.Parameterized):
    boosts = pm.List(default=[boost], class_=Boost)
    new_boost = pm.Action(lambda self: self._new_boost())

    def _new_boost(self):
        self.boosts.append(Boost())

    @pm.depends('boosts')
    def view(self):
        return pn.Row(self, pn.Column(*self.boosts))


boost_factory = BoostFactory()

app = pn.Tabs(
    ('Donations', donations.view()),
    ('Token Distribution', tec_distribution.view()),
    ('TEA Token Distribution', tea_distribution.view()),
    ('SME Signal Boost', boost.view()),
    ('Boost Factory', boost_factory.view()),
    active=4,
)


if __name__ == '__main__':
    print(donations)
