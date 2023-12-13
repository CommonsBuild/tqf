import logging

# from fdfpg import app
import panel as pn
import param as pm
from icecream import ic
from tqf.tegr1 import tegr1_app

# from tqf.tegr3 import tegr3_app
from tqf.tqf_math import QuadraticFunding as TQFMath

# from tqf.tegr2 import tegr2_app


tqf_math = TQFMath()


def pn_exception_handler(ex):
    ic('🔥🔥🔥')
    logging.error('Error', exc_info=ex)
    pn.state.notifications.send(
        'Error: %s' % ex, duration=int(10e3), background='black'
    )


pn.extension(
    'tabulator',
    'mathjax',
    exception_handler=pn_exception_handler,
    notifications=True,
)
pn.state.notifications.position = 'top-right'


# Use the following to initialize ipython when serving the app:
# from IPython import start_ipython
# import threading
# def start_ipython_in_thread(namespace):
#     start_ipython(argv=[], user_ns=namespace)
#
#
# # Pass the main thread's namespace to the IPython instance
# ipython_thread = threading.Thread(target=start_ipython_in_thread, args=(globals(),))
# ipython_thread.start()


template = pn.template.VanillaTemplate(title='Tunable Quadratic Funding')


class App(pm.Parameterized):
    def view(self):
        return pn.Tabs(
            (
                'My Grant Rounds',
                pn.Tabs(
                    ('TEGR1', tegr1_app),
                    # ('TEGR2', tegr2_app),
                    # ('TEGR3', tegr3_app),
                    active=0,
                    dynamic=True,
                ),
            ),
            (
                'Research Tools',
                pn.Tabs(
                    # ('Sim Data', None),
                    # ('FDFPG', None),
                    ('TQF Math', tqf_math.view()),
                    active=0,
                    dynamic=True,
                ),
            ),
            active=0,
            dynamic=True,
            tabs_location='left',
        )


template.main.append(App().view())
template.servable()

# tegr1_app.servable()
