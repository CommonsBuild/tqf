import hvplot.pandas
import numpy as np
import pandas as pd
import panel as pn
import param as pm
from icecream import ic


class Donations(pm.Parameterized):
    dataset = pm.DataFrame(default=None, columns={'voter', 'amountUSD'})
    file = pm.FileSelector(
        default='./app/input/vote_coefficients_input.csv',
        path='./app/input/*.csv',
        precedence=0.5,
    )
    page_size = pm.Integer(default=20, bounds=(5, 100))

    def __init__(self, **params):
        super().__init__(**params)
        self.load_file()

    @pm.depends('file', watch=True)
    def load_file(self):
        self.dataset = pd.read_csv(self.file)

    @pm.depends('dataset', 'page_size')
    def view(self):
        view = pn.Param(
            self.param,
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
        return view


donations = Donations(page_size=5)

# pn.widgets.Tabulator.theme = 'simple'
# pn.widgets.Tabulator(df_qf, layout='fit_data_table', page_size=5)


app = pn.panel(donations.view())

if __name__ == '__main__':
    print(donations)
