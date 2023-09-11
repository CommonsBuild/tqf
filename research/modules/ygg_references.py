#!/usr/bin/env python
# coding: utf-8

# ## Tabulator Add Remove Row Reference

# In[2]:


import panel as pn
import pandas as pd
pn.extension('tabulator')

frame1 = pd.DataFrame({"A": [1, 2, 3]})

table = pn.widgets.Tabulator(value=frame1)
button1 = pn.widgets.Button(name="Add row")
button2 = pn.widgets.Button(name="Remove selected rows")


def change_data(_):
    frame2 = pd.DataFrame({"A": [5]})
    table.stream(frame2)
    
def remove_selected_rows(_):
    table.value = table.value.drop(table.selection)


button1.on_click(change_data)
button2.on_click(remove_selected_rows)
pn.Column(pn.Row(button1,button2), table).servable()


# ## Hvplot Line Segments Reference

# In[9]:


import numpy as np
import pandas as pd
import hvplot.pandas
import holoviews as hv

# Sample data
df = pd.DataFrame({
    'x': np.linspace(0, 10, 10),
    'y': np.sin(np.linspace(0, 10, 10))
})

# Extract the default color cycle from hvplot
colors = hv.plotting.util.process_cmap('Category10', ncolors=len(df)-1)

def get_segments(df):
    """Break the data into individual segments."""
    segments = []
    for i in range(len(df) - 1):
        segment = df.iloc[i:i+2]
        segments.append(segment)
    return segments

# Get the individual segments
segments = get_segments(df)

# Plot each segment as a separate line with its own color
plots = [seg.hvplot.line(x='x', y='y', color=colors[i]) for i, seg in enumerate(segments)]

# Overlay all the line segments to create the final plot
final_plot = hv.Overlay(plots)

final_plot


# ## Hvplot Explorer Reference

# In[2]:


import hvplot.pandas

from bokeh.sampledata.penguins import data as df

hvexplorer = hvplot.explorer(df)


# In[3]:


hvexplorer.param.set_param(kind='kde', x='bill_length_mm', y_multi=['bill_depth_mm'], by=['species'])
hvexplorer.labels.title = 'Penguins Scatter'


# In[4]:


hvexplorer


# In[5]:


settings = hvexplorer.settings()
settings


# In[6]:


df.hvplot(**settings)


# In[7]:


hvexplorer.plot_code()


# In[8]:


df.hvplot(by=['species'], kind='scatter', title='Penguins Scatter', x='bill_length_mm', y=['bill_depth_mm'])


# ## Colormaps Reference

# https://holoviews.org/user_guide/Colormaps.html

# https://github.com/holoviz/hvplot/issues/652

# In[1]:


import holoviews as hv
import numpy as np
hv.extension('bokeh')


# In[2]:


hv.plotting.util.list_cmaps(records=True,category='Categorical',reverse=False, provider='bokeh')


# ## Use Langchain for Data Science Help

# The following is an example of using langchain with a pandas dataframe.

# In[23]:


from langchain.agents import create_pandas_dataframe_agent
from langchain.llms import OpenAI

agent = create_pandas_dataframe_agent(OpenAI(temperature=0), df_qf, verbose=False)

command = agent.run("List columns that only have one value. Return a dataframe that removes those columns. Return the python command to achieve this.")


# In[24]:


command


# ## Tabulator Panel Reference

# In[1]:


import datetime as dt
import numpy as np
import pandas as pd
import panel as pn

np.random.seed(7)
pn.extension('tabulator')


# In[2]:


df = pd.DataFrame({
    'int': [1, 2, 3],
    'float': [3.14, 6.28, 9.42],
    'str': ['A', 'B', 'C'],
    'bool': [True, False, True],
    'date': [dt.date(2019, 1, 1), dt.date(2020, 1, 1), dt.date(2020, 1, 10)],
    'datetime': [dt.datetime(2019, 1, 1, 10), dt.datetime(2020, 1, 1, 12), dt.datetime(2020, 1, 10, 13)]
}, index=[1, 2, 3])

df_widget = pn.widgets.Tabulator(df, buttons={'Print': "<i class='fa fa-print'></i>"})
df_widget


# In[3]:


from bokeh.models.widgets.tables import NumberFormatter, BooleanFormatter

bokeh_formatters = {
    'float': NumberFormatter(format='0.00000'),
    'bool': BooleanFormatter(),
}

pn.widgets.Tabulator(df, formatters=bokeh_formatters)


# In[4]:


tabulator_formatters = {
    'float': {'type': 'progress', 'max': 10},
    'bool': {'type': 'tickCross'}
}

pn.widgets.Tabulator(df, formatters=tabulator_formatters)


# In[5]:


from bokeh.models.widgets.tables import CheckboxEditor, NumberEditor, SelectEditor

bokeh_editors = {
    'float': NumberEditor(),
    'bool': CheckboxEditor(),
    'str': SelectEditor(options=['A', 'B', 'C', 'D']),
}

pn.widgets.Tabulator(df[['float', 'bool', 'str']], editors=bokeh_editors)


# In[6]:


tabulator_editors = {
    'int': None,
    'float': {'type': 'number', 'max': 10, 'step': 0.1},
    'bool': {'type': 'tickCross', 'tristate': True, 'indeterminateValue': None},
    'str': {'type': 'list', 'valuesLookup': True},
    'date': 'date',
    'datetime': 'datetime'
}

edit_table = pn.widgets.Tabulator(df, editors=tabulator_editors)

edit_table


# In[7]:


edit_table.on_edit(lambda e: print(e.column, e.row, e.old, e.value))


# In[8]:


custom_df = pd._testing.makeMixedDataFrame().iloc[:3, :]

pn.widgets.Tabulator(custom_df, widths={'index': 70, 'A': 50, 'B': 50, 'C': 70, 'D': 130})


# In[9]:


pn.widgets.Tabulator(custom_df, widths=130)


# In[10]:


pn.widgets.Tabulator(custom_df, widths={'index': '5%', 'A': '15%', 'B': '15%', 'C': '25%', 'D': '40%'}, sizing_mode='stretch_width')


# In[11]:


pn.widgets.Tabulator(custom_df, layout='fit_data', width=400)


# In[12]:


pn.widgets.Tabulator(custom_df, layout='fit_data_stretch', width=400)


# In[13]:


pn.widgets.Tabulator(custom_df, layout='fit_data_table')


# In[14]:


pn.widgets.Tabulator(custom_df, layout='fit_columns', width=650)


# In[15]:


pn.widgets.Tabulator(df.iloc[:, :2], header_align='center', text_align={'int': 'center', 'float': 'left'}, widths=150)


# In[16]:


style_df = pd.DataFrame(np.random.randn(4, 5), columns=list('ABCDE'))
styled = pn.widgets.Tabulator(style_df)


# In[17]:


styled


# In[18]:


def color_negative_red(val):
    """
    Takes a scalar and returns a string with
    the css property `'color: red'` for negative
    strings, black otherwise.
    """
    color = 'red' if val < 0 else 'black'
    return 'color: %s' % color

def highlight_max(s):
    '''
    highlight the maximum in a Series yellow.
    '''
    is_max = s == s.max()
    return ['background-color: yellow' if v else '' for v in is_max]

styled.style.applymap(color_negative_red).apply(highlight_max)

styled


# Themes:
# 
#     'simple'
# 
#     'default'
# 
#     'midnight'
# 
#     'site'
# 
#     'modern'
# 
#     'bootstrap'
# 
#     'bootstrap4'
# 
#     'materialize'
# 
#     'semantic-ui'
# 
#     'bulma'
# 

# In[19]:


pn.widgets.Tabulator.theme = 'simple'


# In[20]:


styled


# In[21]:


pn.widgets.Tabulator(df, theme='bootstrap5', theme_classes=['thead-dark', 'table-sm'])


# In[22]:


sel_df = pd.DataFrame(np.random.randn(3, 5), columns=list('ABCDE'))

select_table = pn.widgets.Tabulator(sel_df, selection=[0, 2])
select_table


# In[23]:


select_table.selection = [1]

select_table.selected_dataframe


# In[24]:


pn.widgets.Tabulator(sel_df, selection=[0, 2], selectable='checkbox')


# In[25]:


select_table = pn.widgets.Tabulator(sel_df, selectable_rows=lambda df: list(range(0, len(df), 2)))
select_table


# In[26]:


def click(event):
    print(f'Clicked cell in {event.column!r} column, row {event.row!r} with value {event.value!r}')

select_table.on_click(click) 
# Optionally we can also limit the callback to a specific column
# select_table.on_click(click, column='A') 


# In[27]:


wide_df = pd._testing.makeCustomDataframe(3, 10, r_idx_names=['index'])

pn.widgets.Tabulator(wide_df, frozen_columns=['index'], width=400)


# In[28]:


date_df = pd._testing.makeTimeDataFrame().iloc[:5, :2]
agg_df = pd.concat([date_df, date_df.median().to_frame('Median').T, date_df.mean().to_frame('Mean').T])
agg_df.index= agg_df.index.map(str)

pn.widgets.Tabulator(agg_df, frozen_rows=[-2, -1], height=200)


# In[29]:


from bokeh.sampledata.periodic_table import elements

periodic_df = elements[['atomic number', 'name', 'atomic mass', 'metal', 'year discovered']].set_index('atomic number')

content_fn = lambda row: pn.pane.HTML(
    f'<iframe src="https://en.wikipedia.org/wiki/{row["name"]}?printable=yes" width="100%" height="200px"></iframe>',
    sizing_mode='stretch_width'
)

periodic_table = pn.widgets.Tabulator(
    periodic_df, height=350, layout='fit_columns', sizing_mode='stretch_width',
    row_content=content_fn, embed_content=True
)

periodic_table


# In[30]:


periodic_table.expanded


# In[31]:


pn.widgets.Tabulator(date_df.iloc[:3], groups={'Group 1': ['A', 'B'], 'Group 2': ['C', 'D']})


# In[32]:


from bokeh.sampledata.autompg import autompg

pn.widgets.Tabulator(autompg, groupby=['yr', 'origin'], height=240)


# In[33]:


# import bokeh
# bokeh.sampledata.download()


# In[34]:


from bokeh.sampledata.population import data as population_data 
pop_df = population_data[population_data.Year == 2020].set_index(['Location', 'AgeGrp', 'Sex'])[['Value']]

pn.widgets.Tabulator(header_align='center', layout='fit_data_table', value=pop_df, hierarchical=True, aggregators={'Sex': 'sum', 'AgeGrp': 'sum'}, height=200, width=800)


# In[35]:


large_df = pd._testing.makeCustomDataframe(100000, 5)
pn.widgets.Tabulator(large_df, pagination='remote', page_size=3)


# In[36]:


medium_df = pd._testing.makeCustomDataframe(1000, 5)
pn.widgets.Tabulator(medium_df, pagination='local', page_size=3)


# In[37]:


filter_table = pn.widgets.Tabulator(pd._testing.makeMixedDataFrame())
filter_table


# In[ ]:




