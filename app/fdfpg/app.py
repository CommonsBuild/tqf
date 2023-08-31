import param as pm
import panel as pn

from ygg_what_i_learned_c import h1, h2, u, sample_p_i_slider, value_functions
from ygg_what_i_learned_d import positive_marginal_value_vote_results, positive_marginal_value, individual_marginal_value, vpfp


individual_contributions_chart = positive_marginal_value.sum(axis=1).hvplot.line(label='Optimal', title="Marginal Value of Private Contributions vs. Optimal") * individual_marginal_value.sum(axis=1).hvplot.line(label='Private Contributions')


pn.Column(
    vpfp.hvplot.line(title="Mean Value to Society per Public Good Given Funding"),#.hvplot.line(y='value', by='public_good', alpha=0.8, line_width=4, x='funding', title='Mean Value to Society per Public Good Given Funding'),
    h1,
    u.hvplot.heatmap(x='citizens', y='public_goods', C='value', title="Value Matrix Given Funding Levels", cmap='Greens', fontscale=1.2, width=1100, height=850, xlabel='Citizen', ylabel='Public Good', clabel='Amount of value produced by public_good p for citizen i.').opts(default_tools=[]),
    individual_contributions_chart,
    positive_marginal_value_vote_results,
).servable()
