import logging

# from fdfpg import app
import panel as pn
import param as pm
from icecream import ic
from tqf.tegr1 import tegr1_app
from tqf.tegr2 import tegr2_app


def pn_exception_handler(ex):
    ic('🔥🔥🔥')
    logging.error('Error', exc_info=ex)
    pn.state.notifications.send(
        'Error: %s' % ex, duration=int(10e3), background='black'
    )


pn.extension(
    'tabulator', 'mathjax', exception_handler=pn_exception_handler, notifications=True
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


class App(pm.Parameterized):
    def view(self):
        return pn.Tabs(
            ('TEGR1', tegr1_app),
            ('TEGR2', tegr2_app),
            ('TEGR3', None),
            ('Sim Data', None),
            ('FDFPG', None),
            active=1,
            dynamic=True,
        )


App().view().servable()
# tegr1_app.servable()
