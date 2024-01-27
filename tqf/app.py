import logging

import panel as pn
import param as pm
from icecream import ic

from tqf.tegr1 import tegr1_app
from tqf.tegr3 import tegr3_app

# from tqf.tegr2 import tegr2_app

pn.serve(
    {
        'TEGR1': lambda: tegr1_app.servable(),
        'TEGR3': lambda: tegr3_app.servable(),
    },
    admin=False,
    show=False,
    port=5006,
)
