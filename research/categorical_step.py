import hvplot.pandas
import numpy as np
import pandas as pd
import panel as pn

df = pd.DataFrame({'x': list('abcd'), 'y': np.random.rand(4)})

chart = df.hvplot.step(x='x', y='y')

view = pn.panel(chart)

view.servable()
