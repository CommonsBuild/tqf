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

# Donations
donations = Donations(
    name="TEGR3 Donations",
    file="tqf/input/tegr3_vote_coefficients-0x0F0b9d9F72C1660905C57864e79CeB409ADa0C9e.csv",
    grant_names_dataset="tqf/input/tegr3_grants.csv",
)

# Donations Dashboard
donations_dashboard = DonationsDashboard(donations=donations)

# TEC Token Distribution
tec_distribution = TokenDistribution(
    file="tqf/input/tegr3_tec_holders.csv", name="TEC Token", logy=True
)

# TEA Token Distribution
tea_distribution = TokenDistribution(
    file="tqf/input/tegr3_tea_holders.csv", name="TEA Credentials", logy=False
)

# TEC Boost
tec_boost = Boost(
    name="TEGR3 TEC Boost",
    distribution=tec_distribution,
    transformation="LogLinear",
    max_boost=8,
    threshold=10,
)

# TEA Boost
tea_boost = Boost(
    name="TEGR3 TEA Boost",
    distribution=tea_distribution,
    transformation="Linear",
    max_boost=8,
    threshold=1,
)

# Boost Factory
boost_factory = BoostFactory(
    name="TEGR3 Boost Factory",
    boosts=[tec_boost, tea_boost],
    boost_factor=8,
    combine_method="product",
)

# Tunable Quadratic Funding
qf = TunableQuadraticFunding(
    donations_dashboard=donations_dashboard,
    boost_factory=boost_factory,
    mechanism="Cluster Mapping",
    matching_pool=50_000,
    matching_percentage_cap=0.15,
)

# Assemble the app with sidebar
tegr3_app = pn.template.MaterialTemplate(
    title="Tunable Quadratic Funding: TEGR3",
    sidebar=[boost.param for boost in boost_factory.boosts]
    + [boost_factory.param]
    + [qf.param],
)

# Add tabs to the main view
tegr3_app.main += [
    pn.Tabs(
        (
            "Charts",
            pn.Column(
                boost_factory.view_boost_outputs_chart,
                qf.view_qf_matching_bar,
            ),
        ),
        (
            "Data",
            qf.view_results,
        ),
        active=0,
    )
]
