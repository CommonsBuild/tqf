#!/usr/bin/env python
# coding: utf-8

# # Feedback from Jeff Emmett. 
# * Investigate Number of donors
# * Describe Donations (Mean, Median, Mode)
# * Explore the above for tegr1

# In[1]:


import param as pm
import numpy as np
import pandas as pd
import panel as pn
import string
import random
import hvplot.pandas
pn.extension('tabulator')
pd.set_option('display.max_columns', 500)


# In[2]:


df = pd.read_csv('output/TEGR1.csv')


# ### Donors Behavior - Donation Frequency

# In[3]:


df.groupby('voter').count()['id'].to_frame().hvplot.hist(ylabel='Donor Count', xlabel='Number of Projects Donated To', title="Number of Donations per Donor Histogram")


# Here is a table of the above to be more precise.

# In[4]:


df.groupby('voter').count()['id'].to_frame(name='number_of_donations').groupby('number_of_donations').size().reset_index(name='unique donor count').sort_values('number_of_donations')


# ### Adding Stats Mean and Mode and Donor Count to The Model

# In[5]:


class TECQFSME(pm.Parameterized):
    boost_factor = pm.Number(1, bounds=(0,4), step=0.1, doc="Multiplicative factor to apply to boosting coefficient.")
    dataset  = pm.DataFrame(columns={'amountUSD', 'projectId', 'voter'}, precedence=-1, doc="Dataset of donations. Must contain amountUSD, projectId, and voter columns.")
    matching_pool = pm.Integer(25_000, bounds=(0, 100_000), step=5_000, doc="Matching pool amount.")
    total_donations = pm.Number(0, constant=True, doc="Summation of amountUSD from donations dataset.")
    total_funding_pool = pm.Number(0, constant=True, doc="Summation of matching_pool and total_donations.")
    allocations  = pm.DataFrame(precedence=-1, doc="Percentages allocation table.")
    results  = pm.DataFrame(precedence=-1, doc="Matched and unmatched funding amounts. Allocation percentages times funding amounts.")
    
    def __init__(self, **params):
        super().__init__(**params)
        self.dataset = self.qf(self.dataset)
        self.update()
        
    @staticmethod
    def qf(df, column_name='amountUSD', new_column_name='quadratic_amount'):
        """
        Takes a specefied column as the donations column. Applies the QF algorithm to produce a new specefied column and intermediate calculation columns.
        """
        df = df.copy(deep=True)
        df[f'{column_name}_allocation'] = df[column_name] / df[column_name].sum()
        df[f'sqrt({column_name})'] = np.sqrt(df[column_name])
        df[f'sum(sqrt({column_name}))'] = df.groupby('projectId')[f'sqrt({column_name})'].transform('sum')
        df[f'sq(sum(sqrt({column_name})))'] = df[f'sum(sqrt({column_name}))'].transform(lambda x: x**2)
        df[f'{new_column_name}_allocation'] = df[f'sq(sum(sqrt({column_name})))'] / df[f'sq(sum(sqrt({column_name})))'].sum()
        df[new_column_name] = df[column_name].sum() * df[f'{new_column_name}_allocation']
        return df
    
    @staticmethod
    def signal_boost_v1(df, boost_factor, boost_column='amountUSD', new_column_name='amount_boosted'):
        """
        Given a dataset and a specefied column, applies the flag boost algorithm.
        Requires that the dataset contain 'tec_token_flag' and 'tea_flag'.
        """
        df['coefficient'] = 1 + boost_factor * (df['tec_tokens_flag'].astype(int) | df['tea_flag'].astype(int))
        df[new_column_name] = df[boost_column] * df['coefficient']
        return df

    
    @pm.depends('boost_factor', 'matching_pool', watch=True)
    def update(self):
        # Update total donations and funding pool
        with pm.edit_constant(self):
            self.total_donations = self.dataset['amountUSD'].sum()
            self.total_funding_pool = self.matching_pool + self.total_donations
        
        with pm.parameterized.batch_call_watchers(self):
            # Generate and apply the signal boosting coefficient
            self.dataset = self.signal_boost_v1(self.dataset, self.boost_factor, boost_column='amountUSD', new_column_name='amount_boosted')

            # Compute the Boosted Allocation
            self.dataset = self.qf(self.dataset, column_name='amount_boosted', new_column_name='quadratic_amount_boosted')

            # Remove the intermediate steps
            # self.dataset = self.dataset[self.dataset.columns[~self.dataset.columns.str.contains('sqrt')]]

            # Create an allocations table that contains allocation percentages grouped and summed by project. 
            allocation_columns = ['projectId'] + list(self.dataset.columns[self.dataset.columns.str.contains('allocation')])
            self.allocations = self.dataset[allocation_columns].groupby('projectId').sum()

            # Generate the unmatched results table by multiplying allocation percentages by total donations
            unmatched_results = self.total_donations * self.allocations
            
            # Generate the matched results table by multiplying allocation percentages by total donations plus matching pool
            matched_results = self.total_funding_pool * self.allocations
            
            # Merge matched and unmatched results
            self.results = unmatched_results.merge(matched_results, left_index=True, right_index=True, suffixes=('_unmatched', '_matched'))
            
            # Sort results by funding amount
            self.results = self.results.sort_values('quadratic_amount_allocation_matched', ascending=False)
            
            # Add some stats to results
            self.results.insert(0, 'Donor Count', df.groupby('projectId')['voter'].nunique())
            self.results.insert(1, 'Mean Donation', df.groupby('projectId')['amountUSD'].mean())
            self.results.insert(2, 'Median Donation', df.groupby('projectId')['amountUSD'].agg(pd.Series.median))

            # Save the boosting percentage stat
            self.results['SME Percentage Boost'] = 100 * ((self.results['quadratic_amount_boosted_allocation_matched'] - self.results['quadratic_amount_allocation_matched']) / self.results['quadratic_amount_allocation_matched'])


# In[6]:


tec_qf_sme = TECQFSME(dataset=df.copy(deep=True))

tec_qf_sme.results.T


# ### Simulating TEGR1 Data for Experimentation
# Here is an experiment with a tunable parameter w. 

# In[7]:


w = 10

exp1_distribution = lambda w: list(np.ones(w)) + [w]


# In[8]:


exp1_distribution(w)


# In[9]:


def mock_tegr1(distribution: list, tec_token_flag: list, tea_flag: list):
    addresses = lambda: ["0x"+"".join(random.choices(string.hexdigits, k=8)) for _ in range(len(distribution))]
    df = pd.DataFrame([distribution, addresses(), addresses()]).T
    df.columns = ['amountUSD', 'projectId', 'voter']
    df['amountUSD'] = df['amountUSD'].astype(float)
    df['tec_tokens_flag'] = tec_token_flag
    df['tea_flag'] = tea_flag
    return df


# In[10]:


df = mock_tegr1(exp1_distribution(2), 0, 0)


# In[11]:


df


# In[12]:


t = TECQFSME(dataset=df, matching_pool=0, boost_factor=0)

t.dataset


# In[13]:


t.results.T

