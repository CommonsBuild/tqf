import pandas as pd
import panel as pn

#
# from tegr1.tegr1_b_interactive_data import tec_qf_sme
# from tegr1.ygg_main_b_sigmoids import Boost, df_voters_merged
# from widgets.f_m_b import QuadraticFunding
#
# tec_qf_sme.view().servable()
#
# qf =  QuadraticFunding()
#
# qf.view().servable()
#
#
# tec_boost = Boost(signal=df_voters_merged['balance_tec'],
#                   transformation='Threshold', logy=False, threshold=10)
#
# tec_boost.view().servable()
#
#
# tea_boost = Boost(signal=df_voters_merged['balance_tea'],
#                   transformation='Threshold', logy=False, threshold=1)
#
# tea_boost.view().servable()
#
from fdfpg import app
from tegr1 import app

# from ygg_what_i_learned_c import h1, h2, sample_p_i_slider, u, value_functions
# from ygg_what_i_learned_d import (
#     individual_marginal_value,
#     positive_marginal_value,
#     positive_marginal_value_vote_results,
#     vpfp,
# )
# individual_contributions_chart = positive_marginal_value.sum(axis=1).hvplot.line(label='Optimal', title="Marginal Value of Private Contributions vs. Optimal") * individual_marginal_value.sum(axis=1).hvplot.line(label='Private Contributions')
#
#
# pn.Column(
#     vpfp.hvplot.line(title="Mean Value to Society per Public Good Given Funding"),#.hvplot.line(y='value', by='public_good', alpha=0.8, line_width=4, x='funding', title='Mean Value to Society per Public Good Given Funding'),
#     h1,
#     u.hvplot.heatmap(x='citizens', y='public_goods', C='value', title="Value Matrix Given Funding Levels", cmap='Greens', fontscale=1.2, width=1100, height=850, xlabel='Citizen', ylabel='Public Good', clabel='Amount of value produced by public_good p for citizen i.').opts(default_tools=[]),
#     individual_contributions_chart,
#     positive_marginal_value_vote_results,
# ).servable()
app.servable()
