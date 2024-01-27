import hvplot.pandas  # noqa
import panel as pn
import param as pm

from tqf.boost import Boost
from tqf.boost_factory import BoostFactory
from tqf.dataset import Donations, TokenDistribution
from tqf.donations_dashboard import DonationsDashboard

# from .outcomes import Outcomes
from tqf.quadratic_funding import TunableQuadraticFunding

pn.extension("tabulator")

# TEGR2 Donations
tegr2_donations = Donations(
    name="TEGR2 Donations",
    file="tqf/input/tegr2_Token_Engineering_squelched_coefficients.csv",
    grant_names_dataset="tqf/input/tegr2_grants.csv",
)

# TEGR2 Donations Dashboard
tegr2_donations_dashboard = DonationsDashboard(donations=tegr2_donations)

# TEGR2 TEC Token Distribution
tegr2_tec_distribution = TokenDistribution(
    file="tqf/input/tegr2_tec_holders.csv", name="TEC Token", logy=True
)

# TEGR2 TEA Token Distribution
tegr2_tea_distribution = TokenDistribution(
    file="tqf/input/tegr2_tea_holders.csv", name="TEA Credentials", logy=False
)

# TEGR2 TEC Boost
tegr2_tec_boost = Boost(
    name="TEGR2 TEC Boost",
    distribution=tegr2_tec_distribution,
    transformation="Threshold",
    threshold=10,
)

# TEGR2 TEA Boost
tegr2_tea_boost = Boost(
    name="TEGR2 TEA Boost",
    distribution=tegr2_tea_distribution,
    transformation="Threshold",
    threshold=1,
)

# TEGR2 Boost Factory
tegr2_boost_factory = BoostFactory(
    name="TEGR2 Boost Factory",
    boosts=[tegr2_tec_boost, tegr2_tea_boost],
)

# TEGR2 Tunable Quadratic Funding
tegr2_qf = TunableQuadraticFunding(
    donations_dashboard=tegr2_donations_dashboard, boost_factory=tegr2_boost_factory
)

template = pn.template.MaterialTemplate(
    title="Tunable Quadratic Funding",
    sidebar=[boost.param for boost in tegr2_boost_factory.boosts]
    + [tegr2_boost_factory.param]
    + [tegr2_qf.param],
)

template.main += [
    tegr2_boost_factory.view_boost_outputs_chart,
    tegr2_qf.view_qf_matching_bar,
]

tegr2_app = template
