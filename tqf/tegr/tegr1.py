import hvplot.pandas  # noqa
import panel as pn
import param as pm

from tqf.boost import Boost
from tqf.boost_factory import BoostFactory
from tqf.dataset import Donations, TokenDistribution
from tqf.donations_dashboard import DonationsDashboard

# from tqf.outcomes import Outcomes
from tqf.quadratic_funding import TunableQuadraticFunding

pn.extension("tabulator")

# Donations
donations = Donations(
    name="TEGR1 Donations", file="tqf/input/tegr1_vote_coefficients_input.csv"
)

# Donations Dashboard
donations_dashboard = DonationsDashboard(donations=donations)

# TEC Token Distribution
tec_distribution = TokenDistribution(
    file="tqf/input/tegr1_tec_holders.csv", name="TEC Token", logy=True
)

# TEA Token Distribution
tea_distribution = TokenDistribution(
    file="tqf/input/tegr1_tea_holders_dune.csv", name="TEA Credentials", logy=False
)

# TEC Boost
tec_boost = Boost(
    name="TEGR1 TEC Boost",
    distribution=tec_distribution,
    transformation="Threshold",
    threshold=10,
)

# TEA Boost
tea_boost = Boost(
    name="TEGR1 TEA Boost",
    distribution=tea_distribution,
    transformation="Threshold",
    threshold=1,
)

# Boost Factory
boost_factory = BoostFactory(
    name="TEGR1 Boost Factory",
    boosts=[tec_boost, tea_boost],
)

# Tunable Quadratic Funding
qf = TunableQuadraticFunding(
    donations_dashboard=donations_dashboard, boost_factory=boost_factory
)

# outcomes = Outcomes(
#     donations_dashboard=donations_dashboard, boost_factory=boost_factory, tqf=qf
# )

# Dashboard
tabs = pn.Tabs(
    ("Donations", pn.Column(donations.view(), donations_dashboard.view())),
    ("TEC Token Boost", tec_boost.view()),
    ("TEA Token Boost", tea_boost.view()),
    ("Boost Factory", boost_factory.view()),
    ("Tunable Quadradic Funding", qf.view()),
    # ("Outcomes", outcomes.view()),
    active=1,
    dynamic=True,
)

tegr1_app = pn.template.BootstrapTemplate(
    title="Tunable Quadratic Funding: TEGR1", sidebar=[]
)

tegr1_app.main += [tabs]
