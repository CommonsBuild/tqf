import holoviews as hv
import numpy as np
import pandas as pd
import panel as pn
import param as pm

hv.opts.defaults(
    hv.opts.Curve(axiswise=True, framewise=True, shared_axes=False),
    hv.opts.Scatter(axiswise=True, framewise=True, shared_axes=False),
    hv.opts.Image(axiswise=True, framewise=True, shared_axes=False),
    hv.opts.Histogram(axiswise=True, framewise=True, shared_axes=False),
)

from .boost import Boost


class BoostFactory(pm.Parameterized):
    boosts = pm.ListSelector(default=[], precedence=-1)
    combine_method = pm.Selector(default='max', objects=['max', 'sum', 'product'])
    boost_outputs = pm.DataFrame(precedence=1)

    # def _on_boost_change(self, event):
    #     self.param.trigger('combine_method')
    #
    # @pm.depends('boosts', watch=True, on_init=True)
    # def _update_watchers(self):
    #     for boost in self.boosts:
    #         boost.param.watch(self._on_boost_change, 'max_boost')

    @pm.depends('combine_method', watch=True, on_init=True)
    def update_boost_outputs(self):
        boost_outputs_list = [boost.distribution.dataset for boost in self.boosts]

        # Merge Boosts
        if boost_outputs_list:
            boost_outputs = boost_outputs_list[0]
            for idx, df in enumerate(boost_outputs_list[1:], start=1):
                boost_outputs = boost_outputs.merge(
                    df, on='address', how='outer', suffixes=('', f'_{idx}')
                )

        # No boosts. Start with empty dataframe
        else:
            boost_outputs = pd.DataFrame(columns=['address', 'balance', 'boost'])

        # Add suffix to boost 0
        boost_outputs.rename(
            {'balance': 'balance_0', 'boost': 'boost_0'}, axis=1, inplace=True
        )

        # Replace nan with 0
        boost_outputs = boost_outputs.fillna(0)

        boost_columns = [c for c in boost_outputs.columns if c.startswith('boost')]

        # Select the max boost
        boost_outputs['max'] = boost_outputs[boost_columns].max(axis=1)

        # Sum boosts
        boost_outputs['sum'] = boost_outputs[boost_columns].sum(axis=1)

        # Indicate if any boosts have been achieved
        boost_outputs['any'] = (
            (boost_outputs[boost_columns] >= 1).any(axis=1).astype(int)
        )

        # Take the product of boosts in such a way that if any boost is achieved then minimum product is 1, 0 otherwise
        boost_outputs['product'] = np.where(
            boost_outputs['any'],
            boost_outputs[boost_columns].clip(lower=1).prod(axis=1),
            0,
        )

        # Set the total boost column as the combine method
        boost_outputs['total_boost'] = boost_outputs[self.combine_method]

        # Sort by Total Boost and secondly by balance_0
        boost_outputs = boost_outputs.sort_values(
            ['total_boost', 'balance_0'], ascending=False
        )

        # Set boost_outputs dataframe on class
        self.boost_outputs = boost_outputs

    @pm.depends('boost_outputs')
    def view_boost_outputs_table(self):
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
    def view_boost_outputs_chart(self):
        if self.combine_method == 'max':
            stacked = False
            y = [c for c in self.boost_outputs.columns if c.startswith('boost')]
        elif self.combine_method == 'sum':
            stacked = True
            y = [c for c in self.boost_outputs.columns if c.startswith('boost')]

        elif self.combine_method == 'product':
            stacked = False
            y = 'product'

        boosts_view = self.boost_outputs.reset_index(drop=True).hvplot.area(
            x='index',
            y=y,
            stacked=stacked,
            height=400,
            responsive=True,
            shared_axes=False,
        )
        return boosts_view

    def view_boosts_list_param(self):
        return pn.panel(self.param['boosts'], expand_button=False)

    def view_boosts(self):
        return pn.Column(*[boost.view() for boost in self.boosts])

    @pm.depends('boost_outputs')
    def view(self):
        return pn.Column(
            self.view_boosts_list_param,
            self.view_boosts,
            self.param['combine_method'],
            self.view_boost_outputs_table,
            self.view_boost_outputs_chart,
        )
