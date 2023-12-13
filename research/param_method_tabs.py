import time

import hvplot.pandas
import numpy as np
import pandas as pd
import panel as pn
import param as pm

pn.extension()
pn.config.throttled = True


class A(pm.Parameterized):
    data = pm.DataFrame()
    x = pm.Integer(10, bounds=(1, 100))

    @pm.depends('x', watch=True, on_init=True)
    def update_data(self):
        self.data = pd.DataFrame({'x': list(range(self.x)), 'y': np.ones(self.x)})

    def view_data(self):
        time.sleep(3)
        return self.data.hvplot.line(x='x', y='y')

    def view(self):
        return pn.Tabs(
            ('Object', self),
            ('Data', pn.param.ParamMethod(self.view_data)),
            dynamic=True,
        )


a = pn.panel(A().view())

a.servable()
