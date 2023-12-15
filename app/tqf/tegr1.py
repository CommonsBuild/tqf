import hvplot.pandas  # noqa
import panel as pn
import param as pm

from .boost import Boost
from .boost_factory import BoostFactory
from .dataset import TEGR1_TEA, TEGR1_TEC, Donations
from .donations_dashboard import DonationsDashboard

# from .outcomes import Outcomes
from .quadratic_funding import TunableQuadraticFunding

pn.extension('tabulator')

# TEGR1 Donations
tegr1_donations = Donations(
    name='TEGR1 Donations', file='app/input/tegr1_vote_coefficients_input.csv'
)

# TEGR1 Donations Dashboard
tegr1_donations_dashboard = DonationsDashboard(donations=tegr1_donations)

# TEGR1 TEC Token Distribution
tegr1_tec_distribution = TEGR1_TEC(name='TEC Token', logy=True)

# TEGR1 TEA Token Distribution
tegr1_tea_distribution = TEGR1_TEA(name='TEA Credentials', logy=False)

# TEGR1 TEC Boost
tegr1_tec_boost = Boost(
    name='TEGR1 TEC Boost',
    distribution=tegr1_tec_distribution,
    transformation='Threshold',
    threshold=10,
)

# TEGR1 TEA Boost
tegr1_tea_boost = Boost(
    name='TEGR1 TEA Boost',
    distribution=tegr1_tea_distribution,
    transformation='Threshold',
    threshold=1,
)

# TEGR1 Boost Factory
tegr1_boost_factory = BoostFactory(
    name='TEGR1 Boost Factory',
    boosts=[tegr1_tec_boost, tegr1_tea_boost],
)

# TEGR1 Tunable Quadratic Funding
tegr1_qf = TunableQuadraticFunding(
    donations_dashboard=tegr1_donations_dashboard, boost_factory=tegr1_boost_factory
)

# TEGR1 Outcomes
# outcomes = Outcomes(
#     donations_dashboard=tegr1_donations_dashboard,
#     boost_factory=tegr1_boost_factory,
#     tqf=tegr1_qf,
# )

# TEGR1 Dashboard
tegr1_app = pn.Tabs(
    ('Donations', pn.Column(tegr1_donations.view(), tegr1_donations_dashboard.view())),
    # ('TEC Token', pn.Row(tegr1_tec_distribution.view)),
    # ('TEA Token', pn.Row(tegr1_tea_distribution.view)),
    ('TEC Token Boost', tegr1_tec_boost.view()),
    ('TEA Token Boost', tegr1_tea_boost.view()),
    ('Boost Factory', tegr1_boost_factory.view()),
    ('Tunable Quadradic Funding', tegr1_qf.view()),
    # ('Outcomes', outcomes.view()),
    active=1,
    dynamic=True,
)
