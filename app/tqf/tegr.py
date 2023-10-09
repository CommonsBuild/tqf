import hvplot.pandas  # noqa
import panel as pn
import param as pm

from .boost import Boost
from .boost_factory import BoostFactory
from .dataset import TEGR1_TEA, TEGR1_TEC, TEGR2_TEA, TEGR2_TEC, Donations
from .donations_dashboard import DonationsDashboard
from .f_m_b import QuadraticFunding as TQFMath
from .quadratic_funding import TunableQuadraticFunding

pn.extension('tabulator')


# tegr1_donations = Donations(file='./app/input/vote_coefficients_input.csv')
#
# # Select which round to load donations for
# tegr1_donations_dashboard = DonationsDashboard(donations=tegr1_donations)
#
# tegr1_tec_distribution = TEGR1_TEC()
# tegr1_tea_distribution = TEGR1_TEA()
#
#
# tegr1_tec_boost = Boost(
#     name='tegr1_tec_boost',
#     input=tegr1_tec_distribution,
#     transformation='Threshold',
#     threshold=10,
#     token_logy=True,
# )
# tegr1_tec_boost.param['input'].objects = [
#     tegr1_tec_distribution,
#     tegr1_tea_distribution,
# ]
#
# tegr1_tea_boost = Boost(
#     name='tegr1_tea_boost',
#     input=tegr1_tea_distribution,
#     transformation='Threshold',
#     threshold=1,
# )
# tegr1_tea_boost.param['input'].objects = [
#     tegr1_tea_distribution,
#     tegr1_tea_distribution,
# ]
#
#
# tegr1_boost_factory = BoostFactory(boost_template=tegr1_tec_boost)
# tegr1_boost_factory.param['boost_template'].objects = [tegr1_tec_boost, tegr1_tea_boost]
# # boost_factory.boosts = [tegr1_tec_boost, tegr1_tea_boost]
#
# tegr1_qf = TunableQuadraticFunding(
#     donations=tegr1_donations, boost_factory=tegr1_boost_factory
# )
#
# tqf_math = TQFMath()
#
#
# tegr1_app = pn.Tabs(
#     ('Donations', pn.Column(tegr1_donations.view(), tegr1_donations_dashboard.view())),
#     (
#         'Token Distribution',
#         pn.Column(
#             tegr1_tec_distribution.view  # , tegr1_tec_distribution.view_distribution
#         ),
#     ),
#     ('TEA Token Distribution', tegr1_tea_distribution.view()),
#     ('Boost Tuning', tegr1_tec_boost.view()),
#     ('Boost Factory', tegr1_boost_factory.view()),
#     ('Tunable Quadradic Funding', tegr1_qf.view()),
#     ('TQF Math', tqf_math.view()),
#     active=5,
#     dynamic=True,
# )
#
# tegr1_boost_factory._new_boost()
# tegr1_boost_factory.boost_template = tegr1_tea_boost
# tegr1_boost_factory._new_boost()


# TEGR2

tegr2_donations = Donations(
    file='./app/input/Token_Engineering_squelched_coefficients.csv'
)

# Select which round to load donations for
tegr2_donations_dashboard = DonationsDashboard(donations=tegr2_donations)

tegr2_tec_distribution = TEGR2_TEC()
tegr2_tea_distribution = TEGR2_TEA()


tegr2_tec_boost = Boost(
    name='tegr2_tec_boost',
    input=tegr2_tec_distribution,
    transformation='Threshold',
    threshold=10,
    token_logy=True,
)

tegr2_tea_boost = Boost(
    name='tegr2_tea_boost',
    input=tegr2_tea_distribution,
    transformation='Threshold',
    threshold=1,
)


tegr2_boost_factory = BoostFactory(boost_template=tegr2_tec_boost)
tegr2_boost_factory.param['boost_template'].objects = [tegr2_tec_boost, tegr2_tea_boost]

tegr2_qf = TunableQuadraticFunding(
    donations=tegr2_donations, boost_factory=tegr2_boost_factory
)

tqf_math = TQFMath()


tegr2_app = pn.Tabs(
    ('Donations', pn.Column(tegr2_donations.view(), tegr2_donations_dashboard.view())),
    (
        'Token Distribution',
        pn.Column(
            tegr2_tec_distribution.view  # , tegr2_tec_distribution.view_distribution
        ),
    ),
    ('TEA Token Distribution', tegr2_tea_distribution.view()),
    ('Boost Tuning', tegr2_tec_boost.view()),
    ('Boost Factory', tegr2_boost_factory.view()),
    ('Tunable Quadradic Funding', tegr2_qf.view()),
    ('TQF Math', tqf_math.view()),
    active=5,
    dynamic=True,
)

tegr2_boost_factory._new_boost()
tegr2_boost_factory.boost_template = tegr2_tea_boost
tegr2_boost_factory._new_boost()