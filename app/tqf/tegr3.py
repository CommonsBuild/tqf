import hvplot.pandas  # noqa
import panel as pn
import param as pm

from .boost import Boost
from .boost_factory import BoostFactory
from .dataset import TEGR3_TEA, TEGR3_TEC, Donations
from .donations_dashboard import DonationsDashboard
from .outcomes import Outcomes
from .quadratic_funding import TunableQuadraticFunding

pn.extension('tabulator')

# TEGR3 Donations
tegr3_donations = Donations(
    name='TEGR3 Donations',
    file='app/input/tegr3-vote_coefficients-0x0F0b9d9F72C1660905C57864e79CeB409ADa0C9e.csv',
)

# TEGR3 Donations Dashboard
tegr3_donations_dashboard = DonationsDashboard(donations=tegr3_donations)

# TEGR3 TEC Token Distribution
tegr3_tec_distribution = TEGR3_TEC(name='TEC Token', logy=True)

# TEGR3 TEA Token Distribution
tegr3_tea_distribution = TEGR3_TEA(name='TEA Credentials', logy=False)

# TEGR3 TEC Boost
tegr3_tec_boost = Boost(
    name='TEGR3 TEC Boost',
    distribution=tegr3_tec_distribution,
    transformation='Threshold',
    threshold=10,
)

# TEGR3 TEA Boost
tegr3_tea_boost = Boost(
    name='TEGR3 TEA Boost',
    distribution=tegr3_tea_distribution,
    transformation='Threshold',
    threshold=1,
)

# TEGR3 Boost Factory
tegr3_boost_factory = BoostFactory(
    name='TEGR3 Boost Factory',
    boosts=[tegr3_tec_boost, tegr3_tea_boost],
)

# TEGR3 Tunable Quadratic Funding
tegr3_qf = TunableQuadraticFunding(
    donations_dashboard=tegr3_donations_dashboard, boost_factory=tegr3_boost_factory
)

# TEGR3 Outcomes
outcomes = Outcomes(
    donations_dashboard=tegr3_donations_dashboard,
    boost_factory=tegr3_boost_factory,
    tqf=tegr3_qf,
)

# TEGR3 Dashboard
tegr3_app = pn.Tabs(
    ('Donations', pn.Column(tegr3_donations.view(), tegr3_donations_dashboard.view())),
    # ('TEC Token', pn.Row(tegr3_tec_distribution.view)),
    # ('TEA Token', pn.Row(tegr3_tea_distribution.view)),
    ('TEC Token Boost', tegr3_tec_boost.view()),
    ('TEA Token Boost', tegr3_tea_boost.view()),
    ('Boost Factory', tegr3_boost_factory.view()),
    ('Tunable Quadradic Funding', tegr3_qf.view()),
    ('Outcomes', outcomes.view()),
    active=1,
    dynamic=True,
)
