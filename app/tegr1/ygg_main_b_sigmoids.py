#!/usr/bin/env python
# coding: utf-8

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
pd.set_option('display.max_columns', 500)


# Utilities.

# In[2]:


def shorten_hashes(df):
    """
    This method shortens addresses in a dataframe for convenience.
    Transforms string columns of a dataframe on values that begin with 0x.
    For any value that begins with 0x in an object column, take only 10 characters.
    """
    df[df.select_dtypes('object').columns] = df.select_dtypes('object').apply(lambda x: np.where(x.str.startswith('0x'), x.str.slice(stop=10), x))
    return df


# Load dataset according to ygg-a. Add an additional step that further reduces columns.

# In[3]:


# Read the Vote Coefficients Inputs Dataset
ic("...Loading Donations dataset...")
df_donations = pd.read_csv('./input/vote_coefficients_input.csv', parse_dates=['last_score_timestamp'])
ic(df_donations.shape)

# Drop Unsuccessful Rows
ic("...Dropping unsuccessful data...")
ic(len(df_donations[df_donations['success']==False]))
df_donations = df_donations[df_donations['success']==True]
ic(df_donations.shape)

# Drop Unecessary Columns
ic("...Dropping Unecessary columns...")
drop_columns=['success', 'status', 'type']
ic(drop_columns)
df_donations = df_donations.drop(drop_columns, axis=1)
ic(df_donations.shape)

# Shorten Hash Values for Easier Reading
ic("...Shortening hashes...")
df_donations = shorten_hashes(df_donations)
ic(df_donations.shape)

# Further drop columns to isolate algorithm environment
ic("...Dropping Unecessary columns...")
drop_columns=['projectId', 'roundId', 'token', 'amount', 'coefficient', 'last_score_timestamp', 'rawScore', 'threshold']
ic(drop_columns)
df_donations = df_donations.drop(drop_columns, axis=1)
ic(df_donations.shape)

# Use applicationId as projectId to make identifying projects easier
ic("...Rename applicationId to projectId...")
df_donations = df_donations.rename({'applicationId':'projectId'},axis=1)
ic(df_donations.shape)


# In[4]:


df_donations


# Total Donation Amounts Per Project.

# In[5]:


df_projects = df_donations.groupby('projectId').agg(
    amountUSD=('amountUSD', 'sum'),
    donations=('amountUSD', 'size'),
    mean=('amountUSD', 'mean'),
    median=('amountUSD', 'median')
)
df_projects


# Total Donation Amounts Per Voter

# In[6]:


df_voters = df_donations.groupby('voter').agg(
    amountUSD=('amountUSD', 'sum'),
    donations=('amountUSD', 'size'),
    mean=('amountUSD', 'mean'),
    median=('amountUSD', 'median')
)
df_voters


# Load TEC Token Dataset.

# In[7]:


# get table of valid tec holders
# extracted from https://dune.com/queries/2457553/4040451
ic("...Loading TEC Token Holders dataset...")
df_tec = pd.read_csv('./input/tec_holders.csv')
ic(df_tec.shape)

# Shorten Hash Values for Easier Reading
ic("...Shortening hashes...")
df_tec = shorten_hashes(df_tec)
ic(df_tec.shape)

# Take the address and balance columns
ic("...Take address and balance...")
df_tec = df_tec[['address', 'balance']]
ic(df_tec.shape)


# In[8]:


df_tec


# Load TEA Credentials Dataset.

# In[9]:


# get table of te academy token holders
# extracted from https://dune.com/queries/2457581
ic("...Loading TEA Credentials dataset...")
df_tea_dune = pd.read_csv('./input/tea_holders_dune.csv')
df_tea_tea = pd.read_excel('./input/tea_holders_tea.xlsx')
ic(df_tea_dune.shape)
ic(df_tea_tea.shape)

# Examine Intersecting Wallets between dune and tea
ic("...Examine Overlap...")
ic(len(set(df_tea_tea['wallet']).intersection(set(df_tea_dune['wallet']))))
ic(len(set(df_tea_tea['wallet']).union(set(df_tea_dune['wallet']))))

# Supplement Dune data with TEA data and drop duplicates
ic("...Leftjoin to Dune Data...")
df_tea = pd.concat([df_tea_dune, df_tea_tea]).drop_duplicates(subset=['wallet'])
ic(df_tea.shape)

# Shorten Hash Values for Easier Reading
ic("...Shortening hashes...")
df_tea = shorten_hashes(df_tea)
ic(df_tea.shape)

# Make a contiguous index
ic("...Resetting index...")
df_tea = df_tea.reset_index(drop=True)
ic(df_tea.shape)

# Fill balance of TEA with 1 for now
ic("...Fill Nan Balance with 1...")
df_tea = df_tea.fillna(1)
ic(df_tea.shape)

# Rename Wallet to Address to be consistent
ic("...Rename Wallet to Address...")
df_tea = df_tea.rename({'wallet':'address'},axis=1)
ic(df_tea.shape)

# Take the address and balance columns
ic("...Take address and balance columns...")
df_tea = df_tea[['address', 'balance']]
ic(df_tea.shape)


# In[10]:


df_tea


# Number of Voters who have TEC Tokens

# In[11]:


ic(len(set(df_donations['voter']).intersection(set(df_tec['address']))))


# Number of Voters who Have TEA Credentials

# In[12]:


ic(len(set(df_donations['voter']).intersection(set(df_tea['address']))))


# Number of Voters who have Both TEC Tokens and TEA Credentials

# In[13]:


ic(len(set(df_donations['voter']).intersection(set(df_tec['address'])).intersection(set(df_tea['address']))))


# In[14]:


df_voters


# Merge the Data Together.

# In[15]:


# Left join the three tables
df_voters_merged = df_voters.reset_index().merge(
    df_tec, left_on='voter', right_on='address',how='left').merge(
    df_tea, left_on='voter', right_on='address',how='left', suffixes=('_tec', '_tea')).drop(columns=['address_tec','address_tea'])

# Replace Nan values with 0
df_voters_merged = df_voters_merged.fillna(0)


# In[16]:


df_voters_merged


# In[17]:


import param as pm
import numpy as np


# In[18]:


class Boost(pm.Parameterized):
    signal = pm.Series(precedence=-1)
    distribution = pm.Series(constant=True, precedence=-1)
    logy = pm.Boolean(False)
    transformation = pm.ObjectSelector('Sigmoid', objects=['Threshold', 'Linear', 'Sigmoid'])
    threshold = pm.Number(100, precedence=-1, bounds=(0, 1000), step=1)
    sigmoid_frequency = pm.Number(1, precedence=-1, bounds=(0.1,5))
    sigmoid_shift = pm.Number(0, precedence=-1, bounds=(-5,5))
    
    def __init__(self, **params):
        super().__init__(**params)
        self.show_transformation_params()
        
    @pm.depends('logy', 'threshold', 'sigmoid_frequency', 'sigmoid_shift', watch=True)
    def update_distribution(self):
        if self.logy:
            signal = np.log(self.signal+1)
            threshold = np.log(self.threshold)
        else:
            signal = self.signal
            threshold = self.threshold
            
        with pm.edit_constant(self): 
            if self.transformation == 'Threshold':
                self.distribution = self._threshold(signal, threshold)
            elif self.transformation == 'Sigmoid':
                self.distribution = self._sigmoid_scale(signal, k=self.sigmoid_frequency, b=self.sigmoid_shift)
            elif self.transformation == 'Linear':
                self.distribution = self._min_max_scale(signal)
            else:
                raise(Exception(f"Unkown Transformation: {self.transformation}"))
        
    @pm.depends('transformation', watch=True)
    def show_transformation_params(self):

        with pm.parameterized.batch_call_watchers(self):
            self.param['threshold'].precedence = -1
            self.param['sigmoid_frequency'].precedence = -1
            self.param['sigmoid_shift'].precedence = -1

            if self.transformation == 'Threshold':
                self.param['threshold'].precedence = 1
                
            if self.transformation == 'Sigmoid':
                self.param['sigmoid_frequency'].precedence = 1
                self.param['sigmoid_shift'].precedence = 1
                
        self.update_distribution()
                
    
    @staticmethod
    def _sigmoid(x, A=1, k=1, b=0):
        return A / (1 + np.exp(-k * (x - b)))
    
    @staticmethod
    def _min_max_scale(signal):
        return pd.Series((signal -signal.min()) /  (signal.max() - signal.min()))

    @staticmethod
    def _threshold(signal, t):
        return (signal >= t).astype(int)
    
    @staticmethod
    def _mean_std_scale(signal):
        return (signal - signal.mean()) / signal.std()
    
    def _sigmoid_scale(self, signal, **params):
        return self._min_max_scale(self._sigmoid(self._mean_std_scale(signal), **params))
    
    def view_distribution(self):
        return self.distribution.sort_values(ascending=False).reset_index(drop=True).hvplot.step()
    
    def view(self):
        return pn.Row(self, self.view_distribution)


# In[19]:


tec_boost = Boost(signal=df_voters_merged['balance_tec'], transformation='Sigmoid', logy=True, sigmoid_frequency=3, sigmoid_shift=1)
tec_boost.view()


# In[20]:


tea_boost = Boost(signal=df_voters_merged['balance_tea'], transformation='Sigmoid', logy=False, threshold=1, sigmoid_frequency=1)
tea_boost.view()


# Applying the new algorithm.

# In[21]:


boost_factor = 1.5
df_voters_merged['balance_tec_sigmoid'] = tec_boost.distribution
df_voters_merged['balance_tea_sigmoid'] = tea_boost.distribution
df_voters_merged['coefficient'] = 1 + boost_factor * (df_voters_merged['balance_tec_sigmoid'] + df_voters_merged['balance_tea_sigmoid'])


# Inspect the SMEs

# In[22]:


df_sme = df_voters_merged[(df_voters_merged['balance_tec']>0) | (df_voters_merged['balance_tea']>0)].sort_values('coefficient', ascending=False)
df_sme


# In[23]:


df_sme[['amountUSD', 'donations']].sum()


# In[24]:


df_sme[['mean', 'coefficient']].mean()


# Combine Voters Dataset with Donations Dataset.

# In[25]:


# Left join the three tables
df_merged = df_donations.merge(df_voters_merged, left_on='voter', right_on='voter',how='left')
df_merged.sample(5)


# In[26]:


df_merged.sort_values('coefficient', ascending=False)

