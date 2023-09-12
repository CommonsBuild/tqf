import hvplot.pandas  # noqa
import panel as pn
import param as pm

from .boost import Boost
from .boost_factory import BoostFactory
from .dataset import TEA, TEC, Donations
from .donations_dashboard import DonationsDashboard
from .quadratic_funding import QuadraticFunding

pn.extension('tabulator')

donations = Donations()

donations_dashboard = DonationsDashboard(donations=donations)

tec_distribution = TEC()
tea_distribution = TEA()


tegr1_tec_boost = Boost(
    input=tec_distribution,
    transformation='Threshold',
    threshold=10,
    token_logy=True,
)
tegr1_tec_boost.param['input'].objects = [tec_distribution, tea_distribution]

tegr1_tea_boost = Boost(
    input=tea_distribution,
    transformation='Threshold',
    threshold=1,
)
tegr1_tea_boost.param['input'].objects = [tea_distribution, tea_distribution]


boost_factory = BoostFactory(template=tegr1_tec_boost)
boost_factory.param['template'].objects = [tegr1_tec_boost, tegr1_tea_boost]

qf = QuadraticFunding(donations=donations)

app = pn.Tabs(
    ('Donations', pn.Column(donations.view(), donations_dashboard.view())),
    ('Quadradic Funding', qf.view()),
    ('Token Distribution', tec_distribution.view()),
    ('TEA Token Distribution', tea_distribution.view()),
    ('SME Signal Boost', tegr1_tec_boost.view()),
    ('Boost Factory', boost_factory.view()),
    active=0,
    dynamic=True,
)


if __name__ == '__main__':
    print(donations)
