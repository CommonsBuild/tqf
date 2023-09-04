import hvplot.pandas
import numpy as np
import pandas as pd
import panel as pn
import param as pm
from icecream import ic

import logging
def exception_handler(ex):
    ic("🔥🔥🔥")
    logging.error("Error", exc_info=ex)
    pn.state.notifications.send('Error: %s' % ex, duration=int(10e3), background='black')
pn.extension(exception_handler=exception_handler, notifications=True)
pn.state.notifications.position = 'top-right'


class Donations(pm.Parameterized):
    file = pm.FileSelector(
        default='./app/input/vote_coefficients_input.csv',
        path='./app/input/*.csv',
        precedence=0.5,
    )
    page_size = pm.Integer(default=20)
    dataset = pm.DataFrame(default=None, columns={'voter', 'amountUSD'}, label="Donations Dataset")

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
        return view


donations = Donations()

app = pn.Column(donations.view)

if __name__ == '__main__':
    print(donations)
