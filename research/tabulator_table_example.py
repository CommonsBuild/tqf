import numpy as np
import pandas as pd
import panel as pn

pn.extension('tabulator')

# Sample DataFrame
df = pd.DataFrame(np.random.randn(30, 30))

# Tabulator Table
table = pn.widgets.Tabulator(df, pagination=None, page_size=10, height=300)

# Display the table
pn.Row(table).servable()
