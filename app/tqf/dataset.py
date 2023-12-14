import json

import pandas as pd
import panel as pn
import param as pm
from bokeh.models import HoverTool, NumeralTickFormatter

pn.extension('tabulator')


class Dataset(pm.Parameterized):
    file = pm.FileSelector(
        default=None,
        path='app/input/*.csv',
    )
    dataset = pm.DataFrame(
        default=None,
    )
    page_size = pm.Integer(default=20, precedence=-1)

    def __init__(self, **params):
        super().__init__(**params)
        self.load_file()

    @pm.depends('file', on_init=True, watch=True)
    def load_file(self):
        self.dataset = pd.read_csv(self.file)

    @pm.depends('dataset', 'page_size')
    def view_table(self):
        view = pn.panel(
            pn.Param(
                self.param,
                sort=False,
                widgets={
                    'dataset': {
                        'widget_type': pn.widgets.Tabulator,
                        'layout': 'fit_columns',
                        'page_size': self.page_size,
                        'pagination': None,
                        'height': 400,
                        'header_filters': True,
                        'sizing_mode': 'stretch_width',
                    },
                },
            )
        )
        return view


class Donations(Dataset):
    file = pm.FileSelector(
        # default='app/input/tegr1_vote_coefficients_input.csv',
        path='app/input/*.csv',
        constant=True,
    )
    grant_names_dataset = pm.FileSelector(
        # default='app/input/tegr1_grants.csv',
        path='app/input/*grants.csv',
        constant=True,
    )
    dataset = pm.DataFrame(
        default=None,
        columns={'voter', 'amountUSD', 'grantAddress'},
        label='Donations Dataset',
        precedence=1,
    )

    @pm.depends('dataset', 'grant_names_dataset', watch=True, on_init=True)
    def add_grant_names(self):
        if 'Grant Name' not in self.dataset.columns:
            grant_names = pd.read_csv(self.grant_names_dataset)
            self.dataset = pd.merge(
                self.dataset,
                grant_names,
                how='left',
                left_on='grantAddress',
                right_on='Grant Address',
            )

    def view(self):
        return self.view_table()


class TokenDistribution(Dataset):
    file = pm.FileSelector(
        default=None,
        path='app/input/*.csv',
        constant=True,
    )
    dataset = pm.DataFrame(
        default=None, columns={'address', 'balance'}, label='Token Dataset'
    )
    logy = pm.Boolean(True, precedence=-1)

    @pm.depends('dataset', 'logy')
    def view_distribution(self):
        # Use the Bokeh Hover Tool to show formatted numbers in the hover tooltip for balances
        hover = HoverTool(
            tooltips=[('address', '@address'), ('balance', '@balance{0.00}')]
        )

        # Plot a scatter plot of TEC balances on a logy scale.
        distribution_view = (
            self.dataset.sort_values('balance', ascending=False)
            .reset_index(drop=True)
            .hvplot.scatter(
                y='balance',
                x='index',
                yformatter=NumeralTickFormatter(format='0,0'),
                alpha=0.8,
                logy=self.logy,
                hover_cols=['address', 'balance'],
                title=self.name,
                tools=[hover],
                size=800,
                line_width=2,
                height=400,
                responsive=True,
                xlim=(-1, len(self.dataset)),
                color='white',
                line_color='skyblue',
                xlabel='token_holders',
                shared_axes=False,
            )
        )

        return distribution_view

    def view(self):
        return pn.Column(self.view_table(), self.view_distribution())


class TEGR1_TEC(TokenDistribution):
    file = pm.FileSelector(
        default='app/input/tegr1_tec_holders.csv',
        path='app/input/*.csv',
        constant=True,
    )


class TEGR1_TEA(TokenDistribution):
    file = pm.FileSelector(
        default='app/input/tegr1_tea_holders_dune.csv',
        path='app/input/*.csv',
        constant=True,
    )

    @pm.depends('file', watch=True)
    def load_file(self):
        """
        TEA Data might have 'wallet' field. If it does we remap it to 'address'.
        It's also not sorted so we sort it.
        """
        df = pd.read_csv(self.file)
        df.rename({'wallet': 'address'}, axis=1, inplace=True)
        self.dataset = df.sort_values('balance', ascending=False).reset_index(drop=True)


class TEGR2_TEC(TokenDistribution):
    file = pm.FileSelector(
        default='app/input/tegr2_tec_holders.csv',
        path='app/input/*.csv',
        constant=True,
    )


class TEGR2_TEA(TokenDistribution):
    file = pm.FileSelector(
        default='app/input/tegr2_tea_holders.csv',
        path='app/input/*.csv',
        constant=True,
    )

    @pm.depends('file', watch=True)
    def load_file(self):
        """
        TEA Data might have 'wallet' field. If it does we remap it to 'address'.
        """
        df = pd.read_csv(self.file)
        df.rename({'wallet': 'address'}, axis=1, inplace=True)
        self.dataset = df.sort_values('balance', ascending=False).reset_index(drop=True)


class TEGR3_TEC(TEGR2_TEC):
    file = pm.FileSelector(
        default='app/input/tegr3_tec_holders.csv',
        path='app/input/*.csv',
        constant=True,
    )


class TEGR3_TEA(TEGR2_TEA):
    file = pm.FileSelector(
        default='app/input/tegr3_tea_holders.csv',
        path='app/input/*.csv',
        constant=True,
    )

    @pm.depends('file', watch=True)
    def load_file(self):
        """
        TEA Data might have 'wallet' field. If it does we remap it to 'address'.
        """
        df = pd.read_csv(self.file)
        df.rename({'wallet': 'address'}, axis=1, inplace=True)
        self.dataset = df.sort_values('balance', ascending=False).reset_index(drop=True)
