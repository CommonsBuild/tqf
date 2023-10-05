import logging

# from fdfpg import app
import panel as pn
from icecream import ic
from tqf.tegr import tegr1_app as app


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


app.servable()
