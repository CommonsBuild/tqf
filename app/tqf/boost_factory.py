import pandas as pd
import panel as pn
import param as pm

from .boost import Boost


class BoostFactory(pm.Parameterized):
    boosts = pm.ListSelector(default=[], precedence=1)
    combine_method = pm.Selector(default='max', objects=['max', 'sum'])
    boost_outputs = pm.DataFrame(precedence=1)

    @pm.depends('combine_method', 'boosts', watch=True, on_init=True)
    def update_boost_outputs(self):
        boost_outputs_list = [boost.distribution.dataset for boost in self.boosts]

        # No boosts. Return empty dataframe
        if not boost_outputs_list:
            boost_outputs = pd.DataFrame(columns=['address', 'balance', 'boost'])

        # Merge Boosts
        else:
            boost_outputs = boost_outputs_list[0]
            for idx, df in enumerate(boost_outputs_list[1:], start=1):
                boost_outputs = boost_outputs.merge(
                    df, on='address', how='outer', suffixes=('', f'_{idx}')
                )

        # Generate the total boost column depending on combination method
        boost_outputs = boost_outputs.fillna(0)
        if self.combine_method == 'max':
            boost_outputs['total_boost'] = boost_outputs[
                [col for col in boost_outputs.columns if 'boost' in col]
            ].max(axis=1)
        elif self.combine_method == 'sum':
            boost_outputs['total_boost'] = boost_outputs[
                [col for col in boost_outputs.columns if 'boost' in col]
            ].sum(axis=1)

        # Add suffix to boost 0
        boost_outputs = boost_outputs.rename(
            {'balance': 'balance_0', 'boost': 'boost_0'}, axis=1
        )

        # Add Sort primarly by Total Boost and Secondarly by balance 0
        boost_outputs = boost_outputs.sort_values(
            ['total_boost', 'balance_0'], ascending=False
        )
        self.boost_outputs = boost_outputs

    @pm.depends('boost_outputs')
    def view_boost_outputs(self):
        boost_outputs_view = pn.panel(
            pn.Param(
                self.param['boost_outputs'],
                widgets={
                    'boost_outputs': {
                        'widget_type': pn.widgets.Tabulator,
                        'layout': 'fit_columns',
                        'pagination': None,
                        'height': 400,
                        'header_filters': True,
                        'sizing_mode': 'stretch_width',
                    },
                },
            )
        )

        return boost_outputs_view

    @pm.depends('boost_outputs')
    def view_boosts(self):
        if self.combine_method == 'max':
            stacked = False
        elif self.combine_method == 'sum':
            stacked = True

        boosts_view = self.boost_outputs.reset_index(drop=True).hvplot.area(
            x='index',
            y=[c for c in self.boost_outputs.columns if c.startswith('boost')],
            stacked=stacked,
            height=400,
            responsive=True,
        )
        return boosts_view

    def view(self):
        return pn.Column(
            pn.panel(self.param['boosts'], expand_button=False),
            self.param['combine_method'],
            self.view_boost_outputs(),
            self.view_boosts,
        )
