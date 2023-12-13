import hvplot.pandas  # noqa
import panel as pn
import param as pm

from .boost import Boost
from .boost_factory import BoostFactory
from .dataset import TEGR2_TEA, TEGR2_TEC, Donations
from .donations_dashboard import DonationsDashboard

# from .outcomes import Outcomes
from .quadratic_funding import TunableQuadraticFunding

pn.extension('tabulator')

# TEGR2 Donations
tegr2_donations = Donations(
    name='TEGR2 Donations',
    file='app/input/tegr2_Token_Engineering_squelched_coefficients.csv',
    grant_names_dataset='app/input/tegr2_grants.csv',
)

# TEGR2 Donations Dashboard
tegr2_donations_dashboard = DonationsDashboard(donations=tegr2_donations)

# TEGR2 TEC Token Distribution
tegr2_tec_distribution = TEGR2_TEC(name='TEC Token', logy=True)

# TEGR2 TEA Token Distribution
tegr2_tea_distribution = TEGR2_TEA(name='TEA Credentials', logy=False)

# TEGR2 TEC Boost
tegr2_tec_boost = Boost(
    name='TEGR2 TEC Boost',
    distribution=tegr2_tec_distribution,
    transformation='Threshold',
    threshold=10,
)

# TEGR2 TEA Boost
tegr2_tea_boost = Boost(
    name='TEGR2 TEA Boost',
    distribution=tegr2_tea_distribution,
    transformation='Threshold',
    threshold=1,
)

# TEGR2 Boost Factory
tegr2_boost_factory = BoostFactory(
    name='TEGR2 Boost Factory',
    boosts=[tegr2_tec_boost, tegr2_tea_boost],
)

# TEGR2 Tunable Quadratic Funding
tegr2_qf = TunableQuadraticFunding(
    donations_dashboard=tegr2_donations_dashboard, boost_factory=tegr2_boost_factory
)

# TEGR2 Outcomes
# outcomes = Outcomes(
#     donations_dashboard=tegr2_donations_dashboard,
#     boost_factory=tegr2_boost_factory,
#     tqf=tegr2_qf,
# )

# donations_tab = pn.param.ParamFunction(
#     lambda: pn.Column(tegr2_donations.view, tegr2_donations_dashboard.view),
#     lazy=True,
#     name='Donations',
# )
# tec_token_boost_tab = pn.param.ParamMethod(
#     tegr2_tec_boost.view, lazy=True, name='TEC Token Boost'
# )
# tea_token_boost_tab = pn.param.ParamMethod(
#     tegr2_tea_boost.view, lazy=True, name='TEA Token Boost'
# )
# boost_factory_tab = pn.param.ParamMethod(
#     tegr2_boost_factory.view, lazy=True, name='Boost Factory'
# )
# tqf_tab = pn.param.ParamMethod(tegr2_qf.view, lazy=True, name='TQF')
# # outcomes_tab = pn.param.ParamMethod(outcomes.view, lazy=True, name='Outcomes')
#
# tegr2_app = pn.Tabs(
#     donations_tab,
#     tec_token_boost_tab,
#     tea_token_boost_tab,
#     boost_factory_tab,
#     tqf_tab,
#     # outcomes_tab,
#     dynamic=True,
#     active=1,
# )


# TEGR2 Dashboard
# tegr2_app = pn.Tabs(
#     ('Donations', tegr2_donations_dashboard.view()),
#     # ('TEC Token', pn.Row(tegr2_tec_distribution.view)),
#     # ('TEA Token', pn.Row(tegr2_tea_distribution.view)),
#     ('TEC Token Boost', tegr2_tec_boost.view()),
#     ('TEA Token Boost', tegr2_tea_boost.view()),
#     ('Boost Factory', tegr2_boost_factory.view()),
#     ('Tunable Quadradic Funding', tegr2_qf.view()),
#     # ('Outcomes', outcomes.view()),
#     active=0,
#     dynamic=True,
# )

# tegr2_app = pn.Column(tegr2_boost_factory.view(), tegr2_qf.view())

template = pn.template.MaterialTemplate(
    title='Tunable Quadratic Funding',
    sidebar=[boost.param for boost in tegr2_boost_factory.boosts]
    + [tegr2_boost_factory.param]
    + [tegr2_qf.param],
)

# template.main += [boost.view_boost for boost in tegr2_boost_factory.boosts]
# template.main += [tegr2_boost_factory.boost_outputs]
template.main += [
    tegr2_boost_factory.view_boost_outputs_chart,
    # tegr2_qf.view_results_bar,
    tegr2_qf.view_qf_matching_bar,
    # tegr2_qf.results,
]

tegr2_app = template
