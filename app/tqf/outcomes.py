import numpy as np
import pandas as pd
import panel as pn
import param as pm



class Outcomes(pm.Parameterized):

    tqf = pm.Selector()

    def view_boosts(self):
        return self.tqf.boosts

    def view(self):
        view = pn.Row(
            pn.Column(
                self.tqf.donations.param['file'],
                self.tqf.param['boost_coefficient'],
                self.tqf.param['matching_pool'],
                self.tqf.param['matching_percentage_cap'],
            ),
            pn.Column(
                self.view_boosts,
                self.tqf.view_qf_bar(),
                self.tqf.view_qf_distribution_bar(),
                self.tqf.view_qf_matching_bar(),
            )
        )
       # Append a layout to the main area, to demonstrate the list-like API
        return view
