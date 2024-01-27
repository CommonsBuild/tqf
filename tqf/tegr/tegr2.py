import hvplot.pandas  # noqa
import panel as pn
import param as pm

from tqf.boost import Boost
from tqf.boost_factory import BoostFactory
from tqf.dataset import Donations, TokenDistribution
from tqf.donations_dashboard import DonationsDashboard
from tqf.quadratic_funding import TunableQuadraticFunding

pn.extension("tabulator")

# Donations
donations = Donations(
    name="TEGR2 Donations",
    file="tqf/input/tegr2_Token_Engineering_squelched_coefficients.csv",
    grant_names_dataset="tqf/input/tegr2_grants.csv",
)

# Donations Dashboard
donations_dashboard = DonationsDashboard(donations=donations)

# TEC Token Distribution
tec_distribution = TokenDistribution(
    file="tqf/input/tegr2_tec_holders.csv", name="TEC Token", logy=True
)

# TEA Token Distribution
tea_distribution = TokenDistribution(
    file="tqf/input/tegr2_tea_holders.csv", name="TEA Credentials", logy=False
)

# TEC Boost
tec_boost = Boost(
    name="TEGR2 TEC Boost",
    distribution=tec_distribution,
    transformation="Threshold",
    threshold=10,
)

# TEA Boost
tea_boost = Boost(
    name="TEGR2 TEA Boost",
    distribution=tea_distribution,
    transformation="Threshold",
    threshold=1,
)

# Boost Factory
boost_factory = BoostFactory(
    name="TEGR2 Boost Factory",
    boosts=[tec_boost, tea_boost],
)

# Tunable Quadratic Funding
qf = TunableQuadraticFunding(
    donations_dashboard=donations_dashboard, boost_factory=boost_factory
)

template = pn.template.MaterialTemplate(
    title="Tunable Quadratic Funding",
    sidebar=[boost.param for boost in boost_factory.boosts]
    + [boost_factory.param]
    + [qf.param],
)

template.main += [
    boost_factory.view_boost_outputs_chart,
    qf.view_qf_matching_bar,
]

tegr2_app = template
