import hvplot.pandas
import numbergen as ng
import numpy as np
import pandas as pd
import panel as pn
import param

pd.DataFrame([s.param.values() for s in value_functions])

sample_p_i_slider = pn.widgets.IntSlider(
    name='Utility Value Function', start=0, end=len(value_functions) - 1
)

pn.Row(
    sample_p_i_slider,
    pn.bind(lambda i: value_functions[i].view(), i=sample_p_i_slider),
)

df_value_functions = pd.DataFrame([s.f(s.x()) for s in value_functions])
df_value_functions = df_value_functions.T
df_value_functions.shape

df_value_functions.index = np.linspace(0, 1, len(df_value_functions))
df_value_functions.index.name = 'funding'

df_value_functions.columns = [(p, i) for p in public_goods for i in society]
df_value_functions.columns.name = 'value_p_i'

df_value_functions

df_value_functions_melted = df_value_functions.melt(ignore_index=False)
df_value_functions_melted['public_good'] = (
    df_value_functions_melted['value_p_i']
    .astype(str)
    .apply(eval)
    .apply(lambda x: x[0])
    .astype(str)
)
df_value_functions_melted['citizen'] = (
    df_value_functions_melted['value_p_i']
    .astype(str)
    .apply(eval)
    .apply(lambda x: x[1])
    .astype(str)
)
df_value_functions_melted

vpfp = (
    df_value_functions_melted.groupby(['funding', 'public_good'])[['value']]
    .sum()
    .reset_index()
)

vpfp

vpfp = vpfp.pivot_table(columns='public_good', values='value', index='funding')
vpfp

marginal_value = vpfp.diff().div(vpfp.index.to_series().diff(), axis=0).bfill()
marginal_value

positive_marginal_value = marginal_value.where(marginal_value > 1, 0)
positive_marginal_value

individual_marginal_value = (
    df_value_functions.diff()
    .div(df_value_functions.index.to_series().diff(), axis=0)
    .bfill()
)
individual_marginal_value

positive_individual_marginal_value = individual_marginal_value.where(
    individual_marginal_value > 1, 0
)
positive_individual_marginal_value

individual_marginal_value_melted = positive_individual_marginal_value.melt(
    ignore_index=False
)
individual_marginal_value_melted['public_good'] = (
    individual_marginal_value_melted['value_p_i']
    .astype(str)
    .apply(eval)
    .apply(lambda x: x[0])
    .astype(str)
)
individual_marginal_value_melted['citizen'] = (
    individual_marginal_value_melted['value_p_i']
    .astype(str)
    .apply(eval)
    .apply(lambda x: x[1])
    .astype(str)
)
individual_marginal_value_melted

individual_marginal_value = individual_marginal_value_melted.pivot_table(
    index='funding', columns='public_good', values='value', aggfunc='sum'
)
individual_marginal_value

take_positive_index = lambda marginal_value: marginal_value.apply(
    lambda col: col[col != 0].last_valid_index()
).replace(np.nan, 0)

optimal_funding = take_positive_index(positive_marginal_value)
optimal_funding

private_contributions_funding = take_positive_index(individual_marginal_value)
private_contributions_funding

positive_marginal_value.hvplot.area()

individual_marginal_value.hvplot.area()

positive_marginal_value.sum(axis=1).hvplot.line(
    label='Collective Marginal Value',
    title='Suboptimality of Private Contributions',
) * individual_marginal_value.sum(axis=1).hvplot.line(
    label='Private Marginal Value'
)

individual_marginal_value

vote_results = (
    individual_marginal_value_melted.groupby(['funding', 'public_good'])[
        'value'
    ]
    .median()
    .to_frame()
    .pivot_table(index='funding', columns='public_good', values='value')
)
vote_results

(vote_results > 1).astype(int)

positive_marginal_value_vote_results = positive_marginal_value.sum(
    axis=1
).hvplot.line(label='Optimal', title='Marginal Value of 1p1v vs optimal.') * (
    positive_marginal_value * (vote_results > 1).astype(int)
).sum(
    axis=1
).hvplot.line(
    label='1p1v'
)
