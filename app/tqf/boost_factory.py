import pandas as pd
import panel as pn
import param as pm

from .boost import Boost2


class BoostFactory(pm.Parameterized):
    boost_template = pm.Selector(precedence=-1)
    boosts = pm.List(default=[], class_=Boost2, precedence=-1)
    new_boost = pm.Action(lambda self: self._new_boost())
    remove_boost = pm.Action(lambda self: self._remove_boost())
    combine_method = pm.Selector(default='max', objects=['max', 'sum'])
    boost_outputs = pm.DataFrame()

    @pm.depends('boosts', watch=True, on_init=True)
    def _update_watchers(self):
        for boost in self.boosts:
            boost.param.watch(self._on_boost_change, 'distribution')

    def _on_boost_change(self, event):
        self.param.trigger('combine_method')

    def _new_boost(self):
        self.boosts.append((Boost2(**self.boost_template.param.values())))
        self.param.trigger('boosts')

    def _remove_boost(self):
        if len(self.boosts):
            self.boosts.pop()
            self.param.trigger('boosts')

    @pm.depends('boost_outputs')
    def boosts_view(self):
        return pn.Column(*[boost.view() for boost in self.boosts])

    @pm.depends('combine_method', 'boosts', watch=True, on_init=True)
    def update_boosts(self):
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
            boost_outputs['Total_Boost'] = boost_outputs[
                [col for col in boost_outputs.columns if 'Boost' in col]
            ].max(axis=1)
        elif self.combine_method == 'sum':
            boost_outputs['Total_Boost'] = boost_outputs[
                [col for col in boost_outputs.columns if 'Boost' in col]
            ].sum(axis=1)

        # Add suffix to boost 0
        boost_outputs = boost_outputs.rename(
            {'balance': 'balance_0', 'Boost': 'Boost_0'}, axis=1
        )

        # Add Sort primarly by Total Boost and Secondarly by balance 0
        boost_outputs = boost_outputs.sort_values(
            ['Total_Boost', 'balance_0'], ascending=False
        )
        self.boost_outputs = boost_outputs

    def view(self):
        return pn.Row(self, self.boosts_view, self.boost_outputs)

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
