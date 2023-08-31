import pandas as pd
import panel as pn

from tegr1.tegr1_b_interactive_data import tec_qf_sme
from tegr1.ygg_main_b_sigmoids import Boost, df_voters_merged
from widgets.f_m_b import QuadraticFunding

tec_qf_sme.view().servable()

qf =  QuadraticFunding()

qf.view().servable()


tec_boost = Boost(signal=df_voters_merged['balance_tec'],
                  transformation='Threshold', logy=False, threshold=10)

tec_boost.view().servable()


tea_boost = Boost(signal=df_voters_merged['balance_tea'],
                  transformation='Threshold', logy=False, threshold=1)

tea_boost.view().servable()

