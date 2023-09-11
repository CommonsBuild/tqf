#!/usr/bin/env python
# coding: utf-8

# # Loading the TEGR1 Dataset
# 
# This module produces the TEGR1 dataset. This module is the same as main.ipynb with the following changes:
# * Drop unsuccessful donations
# * Remove columns 'success', 'status', and 'type' as these values do not vary.
# * Shorten hashes to 10 total characters to increase readability
# 
# 
# Dropping unsuccessful rows changes the number of rows in the dataset thus you will notice a difference in the stats at the bottom of this notebook from main.ipynb.

# In[1]:


import pandas as pd
import numpy as np
import panel as pn
import hvplot.pandas
from icecream import ic
from bokeh.models.formatters import NumeralTickFormatter
from bokeh.models import HoverTool
ic.configureOutput(prefix='ic|',outputFunction=print)
pn.extension('tabulator')


# ## Read Input Data
# 
# Vote Coefficients Inputs Dataset

# In[2]:


# Read the Vote Coefficients Inputs Dataset
ic("Loading data...")
df_qf = pd.read_csv('./input/vote_coefficients_input.csv', parse_dates=['last_score_timestamp'])
ic(df_qf.shape)

# Drop Unsuccessful Rows
ic(len(df_qf[df_qf['success']==False]))
ic("Dropping unsuccessful data...")
df_qf = df_qf[df_qf['success']==True]
ic(df_qf.shape)

# Drop Unecessary Columns
drop_columns=['success', 'status', 'type']
ic(drop_columns)
ic("Dropping columns...")
df_qf = df_qf.drop(drop_columns, axis=1)
ic(df_qf.shape)

# Shorten Hash Values for Easier Reading
ic("Shortening hashes...")
df_qf[df_qf.select_dtypes('object').columns] = df_qf.select_dtypes('object').apply(lambda x: np.where(x.str.startswith('0x'), x.str.slice(stop=10), x))

df_qf.head(5)


# Exploring data with Tabulator

# In[3]:


pn.widgets.Tabulator.theme = 'simple'
pn.widgets.Tabulator(df_qf, layout='fit_data_table', page_size=5)


# ### Introducing TE Commons Data

# In[4]:


def shorten_hashes(df):
    df[df.select_dtypes('object').columns] = df.select_dtypes('object').apply(lambda x: np.where(x.str.startswith('0x'), x.str.slice(stop=10), x))
    return df


# In[5]:


# get table of valid tec holders
# extracted from https://dune.com/queries/2457553/4040451
df_tec = shorten_hashes(pd.read_csv('./input/tec_holders.csv'))

df_tec


# Visualize the data on a log scale with pretty blue circles.

# In[6]:


# Use the Bokeh Hover Tool to show formatted numbers in the hover tooltip for balances
hover = HoverTool(tooltips=[("address", "@address"), ("balance", "@balance{0.00}")])

# Plot a scatter plot of TEC balances on a logy scale.
df_tec.hvplot.scatter(
    y='balance', 
    yformatter=NumeralTickFormatter(format='0,0'), 
    alpha=0.8, 
    logy=True, 
    hover_cols=['address', 'balance'],
    title="TEC Token Holders Distribution Log Scale",
    tools=[hover],
    size=200,
    color="white",
    line_color="skyblue",
    xlabel="index",
)


# ### Introducing TE Academy Data

# In[7]:


# get table of te academy token holders
# extracted from https://dune.com/queries/2457581
df_tea_dune = shorten_hashes(pd.read_csv('./input/tea_holders_dune.csv'))
df_tea_tea = shorten_hashes(pd.read_excel('./input/tea_holders_tea.xlsx'))

# Combine
df_tea = pd.concat([df_tea_dune, df_tea_tea]).drop_duplicates(subset=['wallet'])

# Make a contiguous index
df_tea = df_tea.reset_index(drop=True)

# Fill balance of TEA with -1 for now
df_tea = df_tea.fillna(-1)


# In[8]:


len(set(df_tea_tea['wallet']).intersection(set(df_tea_dune['wallet'])))


# In[9]:


len(df_tea_dune), len(df_tea_tea)


# In[10]:


df_tea


# Visualize TEA Credentials with scatter and bar plots.

# In[11]:


df_tea.hvplot.scatter(y='balance', x='index', title="TEA Credentials Balances Scatter Plot", alpha=0.8)


# In[12]:


df_tea.groupby('balance').count().hvplot.bar(y='wallet', title="TEA Credentials Balances Bar Chart", ylabel="Wallet Count", alpha=0.8)


# # Calculate Coefficients

# In[13]:


# Drop unecessary columns
df_coef = df_qf.drop(columns=['roundId', 'threshold', 'token', 'last_score_timestamp'])
df_coef


# In[14]:


# Left join the three tables
df_merged = df_qf.merge(
    df_tec, left_on='voter', right_on='address',how='left').merge(
    df_tea, left_on='voter', right_on='wallet',how='left', suffixes=('_tec', '_tea')).drop(columns=['address','wallet'])
df_merged.sample(5)


# In[15]:


# Replace Nan values with 0
df_merged = df_merged.fillna(0)

# Multiply coefficient by 1.5 if tec_flag or tea_flag = 1
df_merged['coefficient'] = 1 + 0.5 * (df_merged['tec_tokens_flag'].astype(int) | df_merged['tea_flag'].astype(int))
df_merged


# # Statistics

# In[16]:


df_merged = df_merged.replace(0,np.nan)


# In[17]:


# some simple statistics on the left join
df_merged[['id','tec_tokens_flag','tea_flag']].count()


# In[18]:


# count the number of unique voters
df_merged[['voter','tec_tokens_flag','tea_flag']].drop_duplicates().count()


# In[19]:


# count the number of voters that have both tec and tea tokens
df_merged[(df_merged['tec_tokens_flag']==True) & (df_merged['tea_flag']==True)][['voter','tec_tokens_flag','tea_flag']].drop_duplicates().count()


# # The TEGR1 Dataset.

# In[20]:


df_merged.to_csv('output/TEGR1.csv', index=False)

