from typing import List

import holoviews as hv
import numpy as np
import pandas as pd
import panel as pn
import param as pm
from bokeh.models import HoverTool

hv.opts.defaults(
    hv.opts.Curve(axiswise=True, framewise=True, shared_axes=False),
    hv.opts.Scatter(axiswise=True, framewise=True, shared_axes=False),
    hv.opts.Image(axiswise=True, framewise=True, shared_axes=False),
    hv.opts.Histogram(axiswise=True, framewise=True, shared_axes=False),
)

from .boost import Boost


class BoostFactory(pm.Parameterized):
    boosts = pm.ListSelector(precedence=-1)
    combine_method = pm.Selector(default='max', objects=['max', 'sum', 'product'])
    boost_outputs = pm.DataFrame(precedence=-1)
    boost_factor = pm.Number(2, bounds=(0, 10), step=0.1)

    def _on_boost_change(self, event):
        self.param.trigger('combine_method')

    @pm.depends('boosts', watch=True, on_init=True)
    def _update_watchers(self):
        for boost in self.boosts:
            boost.param.watch(
                self._on_boost_change,
                'boost',
            )

    def _get_boost_output(
        self, boosts: List[Boost], combine_method: str, boost_factor: float
    ):
        """
        Given a list of boosts, and combine method, outputs a total boost dataset.
        """
        # Generate a list of boost datasets from the list of boosts.
        boost_distribution_datasets: List[pd.Series] = [boost.boost for boost in boosts]

        # If there are boosts, then merge them all into boost_outputs.
        if boost_distribution_datasets:
            boost_outputs: pd.DataFrame = boost_distribution_datasets[0]
            for idx, df in enumerate(boost_distribution_datasets[1:], start=1):
                boost_outputs: pd.DataFrame = boost_outputs.merge(
                    df, on='address', how='outer', suffixes=('', f'_{idx}')
                )
        # No boosts. Start with empty dataframe
        else:
            boost_outputs = pd.DataFrame(columns=['address', 'boost'])

        # Add suffix to boost 0
        boost_outputs.rename({'boost': 'boost_0'}, axis=1, inplace=True)

        # Replace nan with 0
        boost_outputs: pd.DataFrame = boost_outputs.fillna(0)

        # List of boost columns used for calculations.
        boost_columns: List[str] = [
            c for c in boost_outputs.columns if c.startswith('boost')
        ]

        # Select the single max boost that a citizen has received to represent their total boost
        if combine_method == 'max':
            boost_outputs['max'] = boost_outputs[boost_columns].max(axis=1)

        # Sum boosts to get total boost
        if combine_method == 'sum':
            boost_outputs['sum'] = boost_outputs[boost_columns].sum(axis=1)

        # Indicate if any boosts have been achieved
        if combine_method == 'product':
            boost_outputs['any'] = (
                (boost_outputs[boost_columns] >= 1).any(axis=1).astype(int)
            )

        # Take the product of boosts in such a way that if any boost is achieved, then minimum product is 1, and if no boosts are achieve, then the product is 0.
        if combine_method == 'product':
            boost_outputs['product'] = np.where(
                boost_outputs['any'],
                boost_outputs[boost_columns].clip(lower=1).prod(axis=1),
                0,
            )

        # Set the total boost column as the combine method
        boost_outputs['total_boost'] = boost_factor * boost_outputs[combine_method]

        # Sort Primarily by Total Boost and Secondly by boost_0
        boost_outputs = boost_outputs

        # Return boost outputs dataframe
        return boost_outputs

    @pm.depends('combine_method', 'boost_factor', watch=True, on_init=True)
    def update_boost_outputs(self):
        """
        This function updates the boost_outputs dataframe on the boost factory.
        """
        # Set boost_outputs dataframe on class
        self.boost_outputs = self._get_boost_output(
            self.boosts, self.combine_method, self.boost_factor
        )

    @pn.depends('boost_outputs')
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

    @pn.depends('boost_outputs')
    def view_boost_outputs_chart(self):
        boosts = self.boost_outputs.sort_values(
            ['total_boost', 'boost_0'], ascending=False
        )
        boost_columns = [c for c in boosts.columns if c.startswith('boost')]
        boost_names = [boost.name for boost in self.boosts]
        boosts = boosts.rename(columns=dict(zip(boost_columns, boost_names)))
        if self.combine_method == 'max':
            stacked = False
            y = boost_names
            alpha = 1
            max_y = boosts[y].max().max()
            legend_position = 'top_right'
        elif self.combine_method == 'sum':
            stacked = True
            y = boost_names
            alpha = 1
            max_y = boosts[y].sum(axis=1).max()
            legend_position = 'top_right'
        elif self.combine_method == 'product':
            stacked = True
            boosts['harmonic_mean'] = boosts['product'] / boosts[boost_names].sum(
                axis=1
            )
            # boosts[boost_names] = boosts[boost_names] * boosts['harmonic_mean']
            boosts[boost_names] = boosts[boost_names].multiply(
                boosts['harmonic_mean'], axis=0
            )
            y = boost_names
            alpha = 1
            max_y = boosts[y].sum(axis=1).max()
            legend_position = 'top_right'

        boosts[boost_names] = boosts[boost_names] * self.boost_factor
        max_y = max_y * self.boost_factor

        hover = HoverTool(
            tooltips=[('token holder index', '$x{0,0}'), ('boost', '$y{0,0.0}')]
        )
        boosts_view = (
            boosts.reset_index(drop=True)
            .hvplot.area(
                x='index',
                y=y,
                stacked=stacked,
                shared_axes=False,
                alpha=alpha,
                tools=[hover],
                height=400,
                width=1600,
                ylim=(0, max_y),
                group_label='Boost',
                title='Boost Factory',
                ylabel='Boost',
                xlabel='Boost by Token Holder',
            )
            .opts(
                default_tools=[
                    'pan',
                    'box_zoom',
                    'save',
                    'reset',
                ],
            )
        )
        if legend_position is not None:
            boosts_view = boosts_view.opts(legend_position=legend_position)
        return boosts_view

    def view_boosts_list_param(self):
        return pn.panel(self.param['boosts'], expand_button=False)

    def view_boosts(self):
        return pn.Column(*[boost.view() for boost in self.boosts])

    def view(self):
        view = pn.Column(
            self.view_boosts_list_param(),
            self.view_boosts(),
            self.param['combine_method'],
            # self.view_boost_outputs_table,
            self.view_boost_outputs_chart,
        )
        return view
