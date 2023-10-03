import pandas as pd
import panel as pn
import param as pm

pn.extension('tabulator')


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
                        'sizing_mode': 'stretch_width',
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
        default=None,
        columns={'voter', 'amountUSD', 'grantAddress'},
        label='Donations Dataset',
    )


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
