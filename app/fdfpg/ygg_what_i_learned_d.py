import numpy as np

# Number of Citizens in the Society
N = 30

# Max Community Size
C = 20

# Max Number of Public Goods
P = 10

# Society is a set of citizens
society = list(range(N))

# Community is a random subset of the society. The community size is from 25 up to size of the society.
community = np.random.choice(a=list(society), size=C, replace=False, p=None)

# Public Goods are proposed by community members. Cardinality is from 20 up to size of the community.
public_goods = list(enumerate(np.random.choice(a=list(community), size=P, replace=True, p=None)))

community

public_goods

len(society)

len(community)

len(public_goods)

import param
import numpy as np
import panel as pn
import hvplot.pandas
import pandas as pd

class ConcaveFunctionGenerator(param.Parameterized):
    f0 = param.Number(default=0.2, bounds=(0, 1), doc="Value of f(0)")
    f1 = param.Number(default=0.8, bounds=(0, 1), softbounds=(0, 1), doc="Value of f(1)")
    steepness = param.Number(default=5, bounds=(1, 20), doc="Steepness of the curve")

    def __init__(self, **params):
        super().__init__(**params)
        self._update_f1_bounds()

    @param.depends('f0', watch=True)
    def _update_f1_bounds(self):
        # Clip the value of f1 if it's below f0
        self.f1 = max(self.f0, self.f1)
        
        # Update the lower bound of f1 to be the value of f0
        self.param['f1'].bounds = (self.f0, 1)
        
    def x(self):
        return np.linspace(0, 1, 400)

    @param.depends('f0', 'f1', 'steepness')
    def f(self, x):
        # Using the negative exponential function as a base
        y = 1 - np.exp(-self.steepness * x)
        
        # Adjusting the function to start at f0 and end at f1
        y = self.f0 + (self.f1 - self.f0) * (y - y.min()) / (y.max() - y.min())
        
        return y

    @param.depends('f0', 'f1', 'steepness')
    def view(self):
        x = self.x()
        y = self.f(x)
        df = pd.DataFrame({'x': x, 'y': y})
        return df.hvplot.line(x='x', y='y', ylim=(0, 1.01))

concave_gen = ConcaveFunctionGenerator()
pn.Row(concave_gen.param, concave_gen.view).servable()


ConcaveFunctionGenerator(f0=1,f1=0)

import numbergen as ng
import numpy as np


# For CurveGenerator
def concave_function_parameters_generator():
    return dict(
        f0=ng.BoundedNumber(generator=ng.NormalRandom(mu=0.1, sigma=0.3), bounds=(0,1))(),
        f1=ng.BoundedNumber(generator=ng.NormalRandom(mu=0.5, sigma=0.4), bounds=(0,1))(),
        steepness=ng.UniformRandom(lbound=1, ubound=20)(),
    )

concave_function_parameters_generator()

value_functions = [ConcaveFunctionGenerator(**concave_function_parameters_generator()) for p_i in range(len(public_goods)*len(society))]

import pandas as pd

pd.DataFrame([s.param.values() for s in value_functions])

sample_p_i_slider = pn.widgets.IntSlider(name='Utility Value Function', start=0, end=len(value_functions)-1)

pn.Row(sample_p_i_slider, pn.bind(lambda i: value_functions[i].view(), i=sample_p_i_slider))

df_value_functions = pd.DataFrame([s.f(s.x()) for s in value_functions])
df_value_functions = df_value_functions.T
df_value_functions.shape

df_value_functions.index = np.linspace(0,1,len(df_value_functions))
df_value_functions.index.name = "funding"

df_value_functions.columns = [(p, i) for p in public_goods for i in society]
df_value_functions.columns.name = "value_p_i"

df_value_functions

df_value_functions_melted = df_value_functions.melt(ignore_index=False)
df_value_functions_melted['public_good'] = df_value_functions_melted['value_p_i'].astype(str).apply(eval).apply(lambda x: x[0]).astype(str)
df_value_functions_melted['citizen'] = df_value_functions_melted['value_p_i'].astype(str).apply(eval).apply(lambda x: x[1]).astype(str)
df_value_functions_melted

vpfp = df_value_functions_melted.groupby(['funding', 'public_good'])[['value']].sum().reset_index()

vpfp

vpfp = vpfp.pivot_table(columns='public_good', values='value', index='funding')
vpfp

marginal_value = vpfp.diff().div(vpfp.index.to_series().diff(), axis=0).bfill()
marginal_value

positive_marginal_value = marginal_value.where(marginal_value > 1, 0)
positive_marginal_value

individual_marginal_value = df_value_functions.diff().div(df_value_functions.index.to_series().diff(), axis=0).bfill()
individual_marginal_value

positive_individual_marginal_value = individual_marginal_value.where(individual_marginal_value > 1, 0)
positive_individual_marginal_value

individual_marginal_value_melted = positive_individual_marginal_value.melt(ignore_index=False)
individual_marginal_value_melted['public_good'] = individual_marginal_value_melted['value_p_i'].astype(str).apply(eval).apply(lambda x: x[0]).astype(str)
individual_marginal_value_melted['citizen'] = individual_marginal_value_melted['value_p_i'].astype(str).apply(eval).apply(lambda x: x[1]).astype(str)
individual_marginal_value_melted

individual_marginal_value = individual_marginal_value_melted.pivot_table(index='funding', columns='public_good', values='value', aggfunc='sum')
individual_marginal_value

take_positive_index = lambda marginal_value: marginal_value.apply(lambda col: col[col != 0].last_valid_index()).replace(np.nan, 0)

optimal_funding = take_positive_index(positive_marginal_value)
optimal_funding

private_contributions_funding = take_positive_index(individual_marginal_value)
private_contributions_funding

positive_marginal_value.hvplot.area()

individual_marginal_value.hvplot.area()

positive_marginal_value.sum(axis=1).hvplot.line(label='Collective Marginal Value', title="Suboptimality of Private Contributions") * individual_marginal_value.sum(axis=1).hvplot.line(label='Private Marginal Value')

individual_marginal_value

vote_results = individual_marginal_value_melted.groupby(['funding', 'public_good'])['value'].median().to_frame().pivot_table(index='funding', columns='public_good', values='value')
vote_results

(vote_results > 1).astype(int)

positive_marginal_value_vote_results = positive_marginal_value.sum(axis=1).hvplot.line(label="Optimal", title="Marginal Value of 1p1v vs optimal.") * (positive_marginal_value * (vote_results > 1).astype(int)).sum(axis=1).hvplot.line(label="1p1v")


