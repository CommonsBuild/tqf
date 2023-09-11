import pandas as pd
import numpy as np
import panel as pn
import hvplot.pandas
import holoviews as hv
import param as pm

df = pd.read_csv('output/TEGR1.csv')
df.head()

# Inspect number of unique addresses
df.select_dtypes(include=['object']).nunique()

df[['amountUSD', 'coefficient', 'rawScore']].describe()

df[['balance_tec', 'tec_tokens_flag', 'balance_tea', 'tea_flag']].describe()

hvexplorer = hvplot.explorer(
    df, 
    height=400)
hvexplorer.param.set_param(kind='step', x='index', y_multi=['amountUSD', 'rawScore'], by=[])
hvexplorer.labels.title = 'TEC Quadratic Funding Round #1 Data'
hvexplorer.labels.xlabel = 'Index'
hvexplorer.labels.ylabel = 'USD Amount and Raw Score'
hvexplorer

# hvexplorer.param.set_param(kind='scatter', x='rawScore', y_multi=['amountUSD'], by=['projectId'])
# hvexplorer.labels.xlabel = 'Raw Score'
# hvexplorer.labels.ylabel = 'Amount USD'

df

df['sqrt(amountUSD)'] = np.sqrt(df['amountUSD'])
df['sum(sqrt(amountUSD))'] = df.groupby('projectId')['sqrt(amountUSD)'].transform('sum')
df['sq(sum(sqrt(amountUSD)))'] = df['sum(sqrt(amountUSD))'].transform(lambda x: x**2)
df['quadradic_allocation'] = df['sq(sum(sqrt(amountUSD)))'] / df['sq(sum(sqrt(amountUSD)))'].sum()
df['default_allocation'] = df['amountUSD'] / df['amountUSD'].sum()
# df['quadratic_amount'] = df['amountUSD'].sum() * df['quadradic_allocation']

df

df = pd.read_csv('output/TEGR1.csv')
df

def qf(df, column_name='amountUSD', new_column_name='quadratic_amount'):
    df = df.copy(deep=True)
    df[f'{column_name}_allocation'] = df[column_name] / df[column_name].sum()
    df[f'sqrt({column_name})'] = np.sqrt(df[column_name])
    df[f'sum(sqrt({column_name}))'] = df.groupby('projectId')[f'sqrt({column_name})'].transform('sum')
    df[f'sq(sum(sqrt({column_name})))'] = df[f'sum(sqrt({column_name}))'].transform(lambda x: x**2)
    df[f'{new_column_name}_allocation'] = df[f'sq(sum(sqrt({column_name})))'] / df[f'sq(sum(sqrt({column_name})))'].sum()
    df[new_column_name] = df[column_name].sum() * df[f'{new_column_name}_allocation']
    
    return df

df = qf(df)
df

# Compute the boosted allocation
df['amount_boosted'] = df['amountUSD'] * df['coefficient']
df = qf(df, column_name='amount_boosted', new_column_name='quadratic_amount_boosted')
df

# Examine the project allocations
allocations = df[['projectId'] + list(df.columns[df.columns.str.contains('allocation')])].groupby('projectId').sum().drop('amount_boosted_allocation',axis=1)
allocations

allocations.sum()

donations = df['amountUSD'].sum()
df_donations = allocations * donations

matching_pool = 25_000
funding_pool = matching_pool + donations
df_funding_pool = allocations * funding_pool

results = df_donations.merge(df_funding_pool, left_index=True, right_index=True, suffixes=('_unmatched', '_matched'))
results = results.sort_values('quadratic_amount_allocation_unmatched', ascending=False)

results

results.rename({
    'quadratic_amount_allocation_unmatched': 'QF Unmatched', 
    'quadratic_amount_allocation_matched': 'QF Matched', 
    'quadratic_amount_boosted_allocation_matched': 'QF Matched + SME',
}, axis=1).hvplot.bar(
    y=['QF Unmatched', 'QF Matched', 'QF Matched + SME'],
    rot=45,
    stacked=False,
    title="Adding Expertise into the QF Signal",
).opts(multi_level=False)

results['Percentage Boost'] = 100 * ((results['quadratic_amount_boosted_allocation_matched'] - results['quadratic_amount_allocation_matched']) / results['quadratic_amount_allocation_matched'] + 1)


results.hvplot.bar(
    y='Percentage Boost', 
    color='purple', 
    ylim=(0, 180), 
    yformatter="%.0f%%", 
    yticks=list(range(0,200,20)),
    grid=True,
    height=400,
    rot=45,
    title="SME Boost as % of QF Boost by Project",
) * hv.HLine(100)

results

class TECQFSME(pm.Parameterized):
    boosting = pm.Number(1, bounds=(0,2), step=0.1)
    dataset  = pm.DataFrame(precedence=-1)
    matching_pool = pm.Integer(25_000, bounds=(0, 100_000), step=5_000)
    donations = pm.Number(0, constant=True)
    funding_pool = pm.Number(0, constant=True)
    allocations  = pm.DataFrame(precedence=-1)
    results  = pm.DataFrame(precedence=-1)
    
    def __init__(self, **params):
        super().__init__(**params)  
        self.update()
        
    @staticmethod
    def qf(df, column_name='amountUSD', new_column_name='quadratic_amount'):
        df = df.copy(deep=True)
        df[f'{column_name}_allocation'] = df[column_name] / df[column_name].sum()
        df[f'sqrt({column_name})'] = np.sqrt(df[column_name])
        df[f'sum(sqrt({column_name}))'] = df.groupby('projectId')[f'sqrt({column_name})'].transform('sum')
        df[f'sq(sum(sqrt({column_name})))'] = df[f'sum(sqrt({column_name}))'].transform(lambda x: x**2)
        df[f'{new_column_name}_allocation'] = df[f'sq(sum(sqrt({column_name})))'] / df[f'sq(sum(sqrt({column_name})))'].sum()
        df[new_column_name] = df[column_name].sum() * df[f'{new_column_name}_allocation']
        return df
    
    @pm.depends('boosting', 'matching_pool', watch=True)
    def update(self):
        # Update total donations and funding pool
        with pm.edit_constant(self):
            self.donations = self.dataset['amountUSD'].sum()
            self.funding_pool = self.matching_pool + self.donations
        
        with pm.parameterized.batch_call_watchers(self):
            # Update the Coefficient
            self.dataset['coefficient'] = 1 + self.boosting * (self.dataset['tec_tokens_flag'].astype(int) | self.dataset['tea_flag'].astype(int))

            # Apply the Coefficient
            self.dataset['amount_boosted'] = self.dataset['amountUSD'] * self.dataset['coefficient']

            # Compute the Boosted Allocation
            self.dataset = self.qf(self.dataset, column_name='amount_boosted', new_column_name='quadratic_amount_boosted')

            # Remove the intermediate steps
            self.dataset = self.dataset[self.dataset.columns[~self.dataset.columns.str.contains('sqrt')]]

            # Examine the project allocations
            self.allocations = self.dataset[['projectId'] + list(self.dataset.columns[self.dataset.columns.str.contains('allocation')])].groupby('projectId').sum().drop('amount_boosted_allocation',axis=1)

            # Save the results sort by quadratic funding amounts
            self.results = (self.donations * self.allocations).merge((self.funding_pool * self.allocations), left_index=True, right_index=True, suffixes=('_unmatched', '_matched'))
            self.results = self.results.sort_values('quadratic_amount_allocation_unmatched', ascending=False)

            # Save the boosting percentage stat
            self.results['Percentage Boost'] = 100 * ((self.results['quadratic_amount_boosted_allocation_matched'] - self.results['quadratic_amount_allocation_matched']) / self.results['quadratic_amount_allocation_matched'] + 1)

        
    @pm.depends('dataset')
    def view_expertise_signal(self):
        return self.results.rename({
            'quadratic_amount_allocation_unmatched': 'QF Unmatched', 
            'quadratic_amount_allocation_matched': 'QF Matched', 
            'quadratic_amount_boosted_allocation_matched': 'QF Matched + SME',
        }, axis=1).hvplot.bar(
            y=['QF Unmatched', 'QF Matched', 'QF Matched + SME'],
            rot=45,
            stacked=False,
            title="Adding Expertise into the QF Signal",
        ).opts(multi_level=False, legend_position='top_right')
    
    @pm.depends('dataset')
    def view_percentage_boost(self):
        return self.results.hvplot.bar(
            y='Percentage Boost', 
            color='purple', 
            ylim=(0, 180), 
            yformatter="%.0f%%", 
            yticks=list(range(0,200,20)),
            grid=True,
            height=400,
            rot=45,
            title="SME Boost as % of QF Boost by Project",
        ) * hv.HLine(100)
        
    
    def view(self):
        return pn.Row(self.param, self.view_percentage_boost)

df

tec_qf_sme = TECQFSME(dataset=df.replace(np.nan, 0).copy(deep=True))

tec_qf_sme.view()

tec_qf_sme.view_expertise_signal()
