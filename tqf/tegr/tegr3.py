import hvplot.pandas  # noqa
import panel as pn
import param as pm

from tqf.boost import Boost
from tqf.boost_factory import BoostFactory
from tqf.dataset import TEGR3_TEA, TEGR3_TEC, Donations
from tqf.donations_dashboard import DonationsDashboard
# from .outcomes import Outcomes
from tqf.quadratic_funding import TunableQuadraticFunding

pn.extension('tabulator')

# TEGR3 Donations
tegr3_donations = Donations(
    name='TEGR3 Donations',
    file='tqf/input/tegr3_vote_coefficients-0x0F0b9d9F72C1660905C57864e79CeB409ADa0C9e.csv',
    grant_names_dataset='tqf/input/tegr3_grants.csv',
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
    transformation='LogLinear',
    max_boost=8,
    threshold=10,
)

# TEGR3 TEA Boost
tegr3_tea_boost = Boost(
    name='TEGR3 TEA Boost',
    distribution=tegr3_tea_distribution,
    transformation='Linear',
    max_boost=8,
    threshold=1,
)

# TEGR3 Boost Factory
tegr3_boost_factory = BoostFactory(
    name='TEGR3 Boost Factory',
    boosts=[tegr3_tec_boost, tegr3_tea_boost],
    boost_factor=8,
    combine_method='product',
)

# TEGR3 Tunable Quadratic Funding
tegr3_qf = TunableQuadraticFunding(
    donations_dashboard=tegr3_donations_dashboard,
    boost_factory=tegr3_boost_factory,
    mechanism='Cluster Mapping',
    matching_pool=50_000,
    matching_percentage_cap=0.15,
)

tegr3_app = pn.template.MaterialTemplate(
    title='Tunable Quadratic Funding: TEGR3',
    sidebar=[boost.param for boost in tegr3_boost_factory.boosts]
    + [tegr3_boost_factory.param]
    + [tegr3_qf.param],
)

# template.main += [boost.view_boost for boost in tegr3_boost_factory.boosts]
# template.main += [tegr3_boost_factory.boost_outputs]
tegr3_app.main += [
    pn.Tabs(
        (
            'Charts',
            pn.Column(
                tegr3_boost_factory.view_boost_outputs_chart,
                tegr3_qf.view_qf_matching_bar,
            ),
        ),
        (
            'Data',
            tegr3_qf.view_results,
        ),
        active=1,
    )
]
