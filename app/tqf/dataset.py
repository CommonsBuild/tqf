import json

import pandas as pd
import panel as pn
import param as pm
from bokeh.models import HoverTool

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

    @pm.depends('file', on_init=True, watch=True)
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

    @pm.depends('dataset', watch=True, on_init=True)
    def add_grant_names(self):
        if 'Grant Name' not in self.dataset.columns:
            grant_names = pd.read_csv('./app/input/tegr2_grants.csv')
            print(grant_names)
            self.dataset = pd.merge(
                self.dataset,
                grant_names,
                how='left',
                left_on='grantAddress',
                right_on='Grant Address',
            )


class TokenDistribution(Dataset):
    file = pm.FileSelector(
        default=None,
        path='./app/input/*.csv',
    )
    dataset = pm.DataFrame(
        default=None, columns={'address', 'balance'}, label='Token Dataset'
    )

    def view_distribution(self):
        # Use the Bokeh Hover Tool to show formatted numbers in the hover tooltip for balances
        hover = HoverTool(
            tooltips=[('address', '@address'), ('balance', '@balance{0.00}')]
        )

        # Plot a scatter plot of TEC balances on a logy scale.
        distribution_view = self.donations.dataset.hvplot.scatter(
            y='balance',
            yformatter=NumeralTickFormatter(format='0,0'),
            alpha=0.8,
            logy=True,
            hover_cols=['address', 'balance'],
            title='TEC Token Holders Distribution Log Scale',
            tools=[hover],
            size=200,
            color='white',
            line_color='skyblue',
            xlabel='index',
        )

        return distribution_view


class TEGR1_TEC(TokenDistribution):
    file = pm.FileSelector(
        default='./app/input/tec_holders.csv',
        path='./app/input/*.csv',
    )


class TEGR1_TEA(TokenDistribution):
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


class TEGR2_TEC(TokenDistribution):
    file = pm.FileSelector(
        default='./app/input/tec_holders_tegr2.csv',
        path='./app/input/*.csv',
    )


class TEGR2_TEA(TokenDistribution):
    file = pm.FileSelector(
        default='./app/input/tea_holders_teg2.csv',
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
