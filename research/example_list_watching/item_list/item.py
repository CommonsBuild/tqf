import hvplot.pandas
import numpy as np
import pandas as pd
import panel as pn
import param as pm


class Item(pm.Parameterized):
    value = pm.Integer(default=0, bounds=(0, 1000))
    data = pm.DataFrame()

    def get_data(self, x):
        data = pd.DataFrame({'x': list(range(x)), 'y': np.ones((x))})
        return data

    @pn.depends('value', watch=True, on_init=True)
    def update_data(self):
        self.data = self.get_data(self.value)

    @pn.depends('data')
    def view_data_chart(self):
        return self.data.hvplot.line(x='x', y='y')

    def view(self):
        return pn.Row(self, self.view_data_chart)
