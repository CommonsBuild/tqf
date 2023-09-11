#!/usr/bin/env python
# coding: utf-8

# In[15]:


import pandas as pd
import numpy as np


# In[ ]:





# In[ ]:





# # Read Input Data

# In[16]:


# read an excel file into a pandas dataframe
df_gitcoin = pd.read_csv('./input/vote_coefficients_input.csv')
df_gitcoin.head()


# In[17]:


# get table of valid tec holders
# extracted from https://dune.com/queries/2457553/4040451
df_tec = pd.read_csv('./input/tec_holders.csv')[['address','tec_tokens_flag']]
df_tec.head()


# In[18]:


# get table of te academy token holders
# extracted from https://dune.com/queries/2457581
df_tea_dune = pd.read_csv('./input/tea_holders_dune.csv')[['wallet','tea_flag']]
df_tea_tea = pd.read_excel('./input/tea_holders_tea.xlsx')[['wallet','tea_flag']]

df_tea = pd.concat([df_tea_dune, df_tea_tea]).drop_duplicates(subset=['wallet'])

df_tea.head()


# # Calculate Coefficients

# In[19]:


# left join the three tables
df_merge = df_gitcoin\
    .merge(df_tec, left_on='voter', right_on='address',how='left')\
    .merge(df_tea, left_on='voter', right_on='wallet',how='left')
df_merge.head()


# In[20]:


# QF? any other multiplier etc. aggregate to payout by projectid

# multiply coefficient by 1.5 if tec_flag or tea_flag = 1
df_merge['coefficient'] = np.where(df_merge['tea_flag'] == 1,\
                                    df_merge['coefficient'] * 1.5,\
                                    df_merge['coefficient'])
                                    
df_merge['coefficient'] = np.where((df_merge['tec_tokens_flag'] == 1) & (df_merge['coefficient'] == 1), df_merge['coefficient'] * 1.5, df_merge['coefficient'])

df_merge.head()


# In[21]:


# remove last four columns of the dataframe
df_output = df_merge.iloc[:, :-4]
df_output.head()


# In[22]:


# save the dataframe to a csv file
df_output.to_csv('./output/vote_coefficients_output.csv', index=False)


# # Statistics

# In[23]:


# some simple statistics on the left join
df_merge[['id','tec_tokens_flag','tea_flag']].count()


# In[24]:


# count the number of unique voters
df_merge[['voter','tec_tokens_flag','tea_flag']].drop_duplicates().count()


# In[25]:


# count the number of voters that have both tec and tea tokens
df_merge[(df_merge['tec_tokens_flag']==True) & (df_merge['tea_flag']==True)][['voter','tec_tokens_flag','tea_flag']].drop_duplicates().count()

