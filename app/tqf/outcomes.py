import hvplot.pandas
import numpy as np
import pandas as pd
import panel as pn
import param as pm


class Outcomes(pm.Parameterized):

    donations_dashboard = pm.Selector()
    boost_factory = pm.Selector()
    tqf = pm.Selector()

    def view_boosts(self):
        return self.tqf.boosts

    def view(self):
        view = pn.Row(
            pn.Column(
                self.tqf.param['boost_coefficient'],
                self.tqf.param['matching_pool'],
                self.tqf.param['matching_percentage_cap'],
            ),
            pn.Column(
                # self.boost_factory.view_outcomes(),
                # pd.DataFrame([1, 2, 3]).hvplot.scatter(),
                self.donations_dashboard.view(),
            ),
        )
        # Append a layout to the main area, to demonstrate the list-like API
        return view
