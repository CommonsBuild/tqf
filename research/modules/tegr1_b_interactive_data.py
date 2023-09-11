import param as pm
import panel as pn
import holoviews as hv
import pandas as pd
import numpy as np
import hvplot.pandas


# For Pie Charts
from math import pi
from bokeh.io import show
from bokeh.plotting import figure
from bokeh.transform import cumsum
from bokeh.palettes import Category20

df = pd.read_csv('output/TEGR1.csv')

df

class TECQFSME(pm.Parameterized):
    boost_factor = pm.Number(1, bounds=(0,4), step=0.1, doc="Multiplicative factor to apply to boosting coefficient.")
    dataset  = pm.DataFrame(columns={'amountUSD', 'projectId'}, precedence=-1, doc="Dataset of donations. Must contain amountUSD and projectId columns.")
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
            self.dataset = self.dataset[self.dataset.columns[~self.dataset.columns.str.contains('sqrt')]]

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
            title="SME Boost Funding as % of QF Funding by Project",
        ) * hv.HLine(100)
    
    @pm.depends('dataset')
    def view_sme_signal_boost_by_donation(self):
        return self.dataset.hvplot.area(
            y='amountUSD', y2='amount_boosted', 
            logy=True, color='green', title="Logarithmic SME Signal Boost by Donation", 
            xlabel='index')
            
    
    def view(self):
        return pn.Row(self.param, pn.Column(self.view_sme_signal_boost_by_donation, self.view_percentage_boost))

tec_qf_sme = TECQFSME(dataset=df.copy(deep=True).replace(np.nan, 0))

tec_qf_sme.view()

dataset = tec_qf_sme.dataset

dataset

dataset.hvplot.area(y='amountUSD', y2='amount_boosted', logy=False, color='green', title="SME Signal Boost by Donation", width=1000, height=500, ylim=(0,None), xlabel='index')

dataset.hvplot.area(y='amountUSD', y2='amount_boosted', logy=True, color='green', title="Logarithmic SME Signal Boost by Donation", width=1000, height=500, xlabel='index')

from functools import reduce

# Compute the difference between consecutive indices for each group
dataset['diff'] = dataset.groupby('voter').apply(lambda x: x.index.to_series().diff().fillna(0)).reset_index(level=0, drop=True)

# Use the cumulative sum of these differences as a helper series
dataset['block'] = (dataset['diff'] > 1).groupby(dataset['voter']).cumsum()

# Group by 'Category' and the helper series 'block'
grouped = dataset.groupby(['voter', 'block'])

# Initialize an empty plot
plots = []

# Iterate over each group and create an hvplot visualization
for name, group in grouped:
    plot = group.hvplot.area(label=name[0], y='amountUSD', y2='amount_boosted', logy=True, title="Logarithmic SME Signal Boost by Donation by Donor", width=1000, height=500, xlabel='index')
    plots.append(plot)
    
combined_plot = reduce(lambda x, y: x * y, plots)

dataset.hvplot.step(y='amountUSD', logy=True, color='green', title="Logarithmic SME Signal Boost by Donation Colored by Donor", width=1000, height=500, xlabel='index') * combined_plot

# Compute the difference between consecutive indices for each group
dataset['diff'] = dataset.groupby('projectId').apply(lambda x: x.sort_index().index.to_series().diff().fillna(0)).reset_index(level=0, drop=True)

# Use the cumulative sum of these differences as a helper series
dataset['block'] = (dataset['diff'] > 1).groupby(dataset['projectId']).cumsum()

# Group by 'Category' and the helper series 'block'
grouped = dataset.groupby(['projectId', 'block'])

from functools import reduce

# Initialize an empty plot
plots = []

# Iterate over each group and create an hvplot visualization
for name, group in dataset.sort_values('projectId').reset_index(drop=True).groupby('projectId'):
    plot = group.hvplot.area(
        label=name[0], 
        y='amountUSD', 
        y2='amount_boosted', 
        logy=False, title="SME Signal Boost by Donation by Project", width=1000, height=500, xlabel='index')
    plots.append(plot)
    
combined_plot = reduce(lambda x, y: x * y, plots)

# Display the combined plot
combined_plot


dataset['boosted'] = dataset['amount_boosted'] - dataset['amountUSD']

boosted = dataset[dataset['boosted'] > 0]

from functools import reduce

# Initialize an empty plot
plots = []

# Iterate over each group and create an hvplot visualization
for name, group in boosted.sort_values('projectId').reset_index(drop=True).groupby('projectId', as_index=False):
    plot = group.hvplot.bar(
        label=name[0], 
        y='boosted',
        x='index',
        logy=True, 
        title="Logarithmic SME Signal Boost by Donation by Project", 
        width=1000, height=500, rot=45,
        xlabel='index',
    ).opts(logy=True, ylim=(1e-1, None))
    plots.append(plot)
    
combined_plot = reduce(lambda x, y: x * y, plots)

# Display the combined plot
combined_plot


boosted.sort_values('projectId').rename_axis('Index').reset_index(drop=False).hvplot.bar(
        label=name[0], 
        y='boosted',
        x='Index',
        logy=True, 
        title="Logarithmic SME Signal Boost by Donation by Project", 
        width=1000, height=500, rot=90,
        xlabel='index',
        color='projectId',
        cmap='Category20',
    ).opts(logy=True, ylim=(1e-1, None))

boosted.hvplot.bar(
        label=name[0], 
        y='boosted',
        x='index',
        logy=True, 
        title="Logarithmic SME Signal Boost by Donation by Project", 
        width=1000, height=500, rot=90,
        xlabel='index',
        color='projectId',
        cmap='Category20',
    ).opts(logy=True, ylim=(1e-1, None))

boosted.hvplot.bar(
        label=name[0], 
        y='boosted',
        x='index',
        logy=True, 
        title="Logarithmic SME Signal Boost by Donation by Donor", 
        width=1200, height=600, rot=90,
        xlabel='index',
        color='voter',
        cmap='Category20',
    ).opts(ylim=(1e-1, None))

data = boosted.groupby('voter', sort=False)['boosted'].sum().to_frame(name='Boosted Funding')
data.reset_index().hvplot.bar(
    rot=90,
    title="Directly Boosted Funding Amounts Per Donor",
    height=740,
    width=1100,
    color='voter',
    x='voter',
    cmap='Category20',
    logy=True,
).opts(ylim=(1e-1, None))

data['angle'] = data['Boosted Funding']/data['Boosted Funding'].sum() * 2*pi

# Use the Category20 colormap and cycle through it if there are more than 20 categories
colors = [Category20[20][i % 20] for i in range(len(data))]
data['color'] = colors

p = figure(height=650, title="Donor Boosting Distribution", toolbar_location=None,
           tools="hover", tooltips="Donor: @voter\n Boosted Funding: $@{Boosted Funding}{1,11}", x_range=(-0.5, 1.0))

p.wedge(x=0, y=1, radius=0.4, 
        start_angle=cumsum('angle', include_zero=True), end_angle=cumsum('angle'),
        line_color="white", fill_color='color', legend_field='voter', source=data.reset_index())

p.axis.axis_label = None
p.axis.visible = False
p.grid.grid_line_color = None

show(p)

dataset['Donation Index per Project'] = dataset.groupby('projectId').cumcount() + 1

dataset.hvplot.area(y='amountUSD', y2='amount_boosted',  by='projectId', x='Donation Index per Project', stacked=False, alpha=1, title="SME Signal Boost by Donation by Project", width=1000, height=500, ylim=(0,None)).opts(legend_position='top_right')#.opts(hv.opts.Area(color=hv.Palette('Category20')))

dataset.hvplot.area(y='amountUSD', y2='amount_boosted', by='projectId', x='Donation Index per Project', stacked=False, alpha=1, logy=True, title="Logarithmic SME Signal Boost by Donation by Project", width=1000, height=500).opts(legend_position='top_right')#.opts(hv.opts.Area(color=hv.Palette('Category20')))

dataset['Donation Index per Donor'] = dataset.groupby('voter').cumcount() + 1

dataset.hvplot.area(
    y='amountUSD', y2='amount_boosted',  by='voter', x='Donation Index per Donor', stacked=False, alpha=1, 
    title="SME Signal Boost by Donation by Donor",
    width=1000, height=500, ylim=(0,None)).opts(legend_position='top_right')#.opts(hv.opts.Area(color=hv.Palette('Category20')))

dataset.sort_values('amount_boosted', ascending=False)

dataset.hvplot.area(
    y='amountUSD', y2='amount_boosted',  by='voter', x='Donation Index per Donor', stacked=False, alpha=1, 
    title="Logarithmic SME Signal Boost by Donation by Donor", logy=True,
    width=1000, height=500, ylim=(1e-1,None)).opts(legend_position='top_right')#.opts(hv.opts.Area(color=hv.Palette('Category20')))

boosted.sort_values('amount_boosted', ascending=False).head(20)[['id', 'projectId', 'applicationId', 'voter', 'grantAddress','amountUSD','coefficient','amount_boosted', 'boosted']]

data = boosted.groupby('projectId')['boosted'].sum().to_frame(name='Boosted Funding').sort_values('Boosted Funding', ascending=False)

# Extract the default color cycle from hvplot
default_colors = hv.plotting.util.process_cmap('Category20',  ncolors=len(data))

# Create a color map for the bar chart using the default color cycle
color_map = {cat: default_colors[i] for i, cat in enumerate(data.index)}


data.hvplot.bar(
    rot=90,
    title="Directly Boosted Funding Amounts Per Project",
    height=540,
    width=1100,
).opts(color='projectId', cmap=color_map, legend_position='top_right')

data['angle'] = data['Boosted Funding']/data['Boosted Funding'].sum() * 2*pi

# Use the Category20 colormap and cycle through it if there are more than 20 categories
colors = [Category20[20][i % 20] for i in range(len(data))]
data['color'] = colors

p = figure(height=650, title="Project Boosting Distribution", toolbar_location=None,
           tools="hover", tooltips="Project: @projectId\n Boosted Funding: $@{Boosted Funding}{1,11}", x_range=(-0.5, 1.0))

p.wedge(x=0, y=1, radius=0.4, 
        start_angle=cumsum('angle', include_zero=True), end_angle=cumsum('angle'),
        line_color="white", fill_color='color', legend_field='projectId', source=data.reset_index())

p.axis.axis_label = None
p.axis.visible = False
p.grid.grid_line_color = None

show(p)

data = boosted.groupby('voter')['boosted'].sum().to_frame(name='Boosted Funding').sort_values('Boosted Funding', ascending=False)

# Extract the default color cycle from hvplot
default_colors = hv.plotting.util.process_cmap('Category20',  ncolors=len(data))

# Create a color map for the bar chart using the default color cycle
color_map = {cat: default_colors[i] for i, cat in enumerate(data.index)}


data.hvplot.bar(
    rot=90,
    title="Directly Boosted Funding Amounts Per Donor",
    height=740,
    width=1100,
).opts(color='voter', cmap=color_map, legend_position='top_right')

data['angle'] = data['Boosted Funding']/data['Boosted Funding'].sum() * 2*pi

# Use the Category20 colormap and cycle through it if there are more than 20 categories
colors = [Category20[20][i % 20] for i in range(len(data))]
data['color'] = colors

p = figure(height=650, title="Donor Boosting Distribution", toolbar_location=None,
           tools="hover", tooltips="Donor: @voter\n Boosted Funding: $@{Boosted Funding}{1,11}", x_range=(-0.5, 1.0))

p.wedge(x=0, y=1, radius=0.4, 
        start_angle=cumsum('angle', include_zero=True), end_angle=cumsum('angle'),
        line_color="white", fill_color='color', legend_field='voter', source=data.reset_index())

p.axis.axis_label = None
p.axis.visible = False
p.grid.grid_line_color = None

show(p)

data = dataset[['amountUSD', 'boosted']].sum().to_frame(name='Value').rename_axis('Category').reset_index()
data['Category'] = ['Default Donation Amounts', 'Boost Amounts']
data

data['angle'] = data['Value']/data['Value'].sum() * 2*pi

# Use the Category20 colormap and cycle through it if there are more than 20 categories
colors = [Category20[20][i % 20] for i in range(len(data))]
data['color'] = colors

p = figure(height=650, title="Donations Amounts vs. Boost Amounts", toolbar_location=None,
           tools="hover", tooltips="Category: @Category <br> Value: $@{Value}{1,11}", x_range=(-0.5, 1.0))

p.wedge(x=0, y=1, radius=0.4, 
        start_angle=cumsum('angle', include_zero=True), end_angle=cumsum('angle'),
        line_color="white", fill_color='color', legend_field='Category', source=data.reset_index())

p.axis.axis_label = None
p.axis.visible = False
p.grid.grid_line_color = None

show(p)

tec_qf_sme.allocations

tec_qf_sme.view()

tec_qf_sme.view_expertise_signal()




