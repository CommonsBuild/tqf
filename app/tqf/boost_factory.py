import pandas as pd
import panel as pn
import param as pm

from .boost import Boost


class BoostFactory(pm.Parameterized):
    boosts = pm.ListSelector(default=[], precedence=1)
    # new_boost = pm.Action(lambda self: self._new_boost())
    # remove_boost = pm.Action(lambda self: self._remove_boost())
    combine_method = pm.Selector(default='max', objects=['max', 'sum'])
    boost_outputs = pm.DataFrame(precedence=1)

    # @pm.depends('boosts', watch=True, on_init=True)
    # def _update_watchers(self):
    #     for boost in self.boosts:
    #         boost.param.watch(self._on_boost_change, 'distribution')

    # def _on_boost_change(self, event):
    #     self.param.trigger('combine_method')

    # def _new_boost(self):
    #     self.boosts.append((Boost2(**self.boost_template.param.values())))
    #     self.param.trigger('boosts')
    #
    # def _remove_boost(self):
    #     if len(self.boosts):
    #         self.boosts.pop()
    #         self.param.trigger('boosts')

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
        )
        return boosts_view
        # return pn.Column(*[boost.view_boost() for boost in self.boosts])

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
                self.param,
                widgets={
                    'dataset': {
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

    def view(self):
        return pn.Column(
            self.param['boosts'],
            self.param['combine_method'],
            self.view_boost_outputs(),
            self.view_boosts,
        )

    # @pm.depends('boosts', on_init=True)
    # def view_outcomes(self):
    #     boosts = []
    #
    #     for boost in self.boosts:
    #         boost_view = boost.input.view_distribution
    #         if boost_view:
    #             boosts.append(boost_view)
    #
    #     return pn.Row(*boosts)
