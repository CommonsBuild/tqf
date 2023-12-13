import numpy as np
import pandas as pd
import panel as pn

pn.extension('tabulator')
style_df = pd.DataFrame(np.random.randn(4, 5), columns=list('ABCDE'))
styled = pn.widgets.Tabulator(style_df)


def color_negative_red(val):
    """
    Takes a scalar and returns a string with
    the css property `'color: red'` for negative
    strings, black otherwise.
    """
    color = 'red' if val < 0 else 'black'
    return 'color: %s' % color


def highlight_max(s):
    """
    highlight the maximum in a Series yellow.
    """
    is_max = s == s.max()
    return ['background-color: yellow' if v else '' for v in is_max]


styled.style.applymap(color_negative_red).apply(highlight_max)

styled.servable()
