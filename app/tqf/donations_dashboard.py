import colorcet as cc
import holoviews as hv
import hvplot.networkx as hvnx
import networkx as nx
import numpy as np
import pandas as pd
import panel as pn
import param as pm
from bokeh.models import (
    BasicTicker,
    ColorBar,
    HoverTool,
    LinearColorMapper,
    PrintfTickFormatter,
)
from bokeh.palettes import RdYlGn as bokeh_RdYlGn

RdYlGn = bokeh_RdYlGn[11][::-1]  # This reverses the chosen palette

pn.extension('tabulator')


def color_based_on_eth_address(val):
    # Use the first 6 characters after '0x' for the color
    hex_color = f'#{val[2:8]}'

    # Convert the hex color to an integer to determine if it's light or dark
    bg_color = int(val[2:8], 16)

    # Determine if background color is light or dark for text color
    text_color = 'black' if bg_color > 0xFFFFFF / 2 else 'white'

    # Return the CSS style with the calculated text color
    return f'background-color: {hex_color}; color: {text_color};'


class DonationsDashboard(pm.Parameterized):
    donations = pm.Selector(precedence=-1)

    @pm.depends('donations.dataset')
    def donor_view(self):
        donations_df = self.donations.dataset
        donor_vote_counts = (
            donations_df.groupby('voter')
            .count()['id']
            .to_frame(name='number_of_donations')
        )
        histogram = donor_vote_counts.hvplot.hist(
            ylabel='Donor Count',
            xlabel='Number of Projects Donated To',
            title='Number of Donations per Donor Histogram',
            height=320,
        )
        table = (
            donor_vote_counts.groupby('number_of_donations')
            .size()
            .reset_index(name='unique donor count')
            .sort_values('number_of_donations')
            .hvplot.table(height=320)
        )
        return pn.Row(histogram, table)

    @pm.depends('donations.dataset')
    def sankey_view(self):
        donations_df = self.donations.dataset
        sankey = hv.Sankey(donations_df[['voter', 'Grant Name', 'amountUSD']])
        return sankey

    @pm.depends('donations.dataset')
    def projects_view(self):
        donations_df = self.donations.dataset

        # Calculate Data per Project
        projects = (
            donations_df.groupby('Grant Name')
            .apply(
                lambda group: pd.Series(
                    {
                        'donor_count': group['voter'].nunique(),
                        'mean_donation': group['amountUSD'].mean(),
                        'median_donation': group['amountUSD'].median(),
                        'total_donation': group['amountUSD'].sum(),
                        'max_donation': group['amountUSD'].max(),
                        'max_doner': group.loc[group['amountUSD'].idxmax(), 'voter'],
                        'donations': sorted(group['amountUSD'].tolist(), reverse=True),
                    }
                )
            )
            .reset_index()
        )
        # Format the donations list
        projects['donations'] = projects['donations'].apply(
            lambda donations: ['${:.2f}'.format(n) for n in donations]
        )

        # Use tabulator to display the data
        projects_view = pn.widgets.Tabulator(
            projects,
            formatters={'donations': {'type': 'textarea', 'textAlign': 'left'}},
        )
        projects_view.style.applymap(color_based_on_eth_address, subset='max_doner')
        return projects_view

    @pm.depends('donations.dataset')
    def contributors_view(self):
        """
        Note, the following three terms are conflated: donor, contributor, and voter.
        """
        donations_df = self.donations.dataset

        # Calculate Data per Contributor
        contributors = (
            donations_df.groupby('voter')
            .apply(
                lambda group: pd.Series(
                    {
                        'project_count': group['Grant Name'].nunique(),
                        'mean_donation': group['amountUSD'].mean(),
                        'median_donation': group['amountUSD'].median(),
                        'total_donation': group['amountUSD'].sum(),
                        'max_donation': group['amountUSD'].max(),
                        'max_grant': group.loc[
                            group['amountUSD'].idxmax(), 'Grant Name'
                        ],
                        'donations': sorted(group['amountUSD'].tolist(), reverse=True),
                    }
                )
            )
            .reset_index()
        )

        # Format the donations list
        contributors['donations'] = contributors['donations'].apply(
            lambda donations: ['${:.2f}'.format(n) for n in donations]
        )

        # Use tabulator to display the data
        contributors_view = pn.widgets.Tabulator(
            contributors,
            formatters={'donations': {'type': 'textarea', 'textAlign': 'left'}},
        )
        contributors_view.style.applymap(color_based_on_eth_address, subset='voter')
        return contributors_view

    @pm.depends('donations.dataset', watch=True)
    def contributions_matrix(self):
        donations_df = self.donations.dataset
        contributions_matrix = donations_df.pivot_table(
            index='voter', columns='Grant Name', values='amountUSD', aggfunc='sum'
        )
        return contributions_matrix

    @pm.depends('donations.dataset')
    def contributions_matrix_view(self):
        contributions_matrix = self.contributions_matrix().reset_index()
        contributions_matrix_view = pn.widgets.Tabulator(contributions_matrix)
        contributions_matrix_view.style.applymap(
            color_based_on_eth_address,
            subset='voter',
        )
        return contributions_matrix_view

    @pm.depends('donations.dataset')
    def contributions_network_view(self):
        # Replace nan values with 0
        donations_df = self.donations.dataset.replace(0, np.nan)

        # Explicitly set string columns as str
        donations_df['voter'] = donations_df['voter'].astype(str)
        donations_df['Grant Name'] = donations_df['Grant Name'].astype(str)

        # Create graph from the dataframe
        G = nx.from_pandas_edgelist(
            donations_df,
            'voter',
            'Grant Name',
            ['amountUSD'],
            create_using=nx.Graph(),
        )

        # Modify edge width to be the donation size divided by 10
        for u, v, d in G.edges(data=True):
            d['amountUSD'] = d['amountUSD'] / 40

        # Assigning custom colors per node type
        voter_color_value = 1
        public_good_color_value = 2
        custom_cmap = {
            voter_color_value: 'purple',
            public_good_color_value: 'orange',
            'default': 'lightgray',
        }

        # Calculate the total donation for each public good
        total_donations_per_public_good = donations_df.groupby('Grant Name')[
            'amountUSD'
        ].sum()

        # Find the max and min total donations for normalization
        max_size = total_donations_per_public_good.max()
        min_size = total_donations_per_public_good.min()

        def get_hex_color(address):
            hex_color = f'#{address[2:8]}'
            return hex_color

        # Set node attributes
        for node in G.nodes():
            if node in donations_df['voter'].unique():
                G.nodes[node]['size'] = donations_df[donations_df['voter'] == node][
                    'amountUSD'
                ].sum()
                G.nodes[node]['color'] = voter_color_value
                G.nodes[node]['id'] = node
                G.nodes[node]['shape'] = 'circle'
                G.nodes[node]['type'] = 'voter'
                G.nodes[node]['outline_color'] = get_hex_color(node)
            else:
                G.nodes[node]['size'] = donations_df[
                    donations_df['Grant Name'] == node
                ]['amountUSD'].sum()
                G.nodes[node]['id'] = node
                G.nodes[node]['shape'] = 'triangle'
                G.nodes[node]['type'] = 'public_good'
                G.nodes[node]['outline_color'] = 'black'  # Outline color for voters
                G.nodes[node]['color'] = public_good_color_value

        # Now, calculate max_size and min_size based on the node attributes
        public_goods_sizes = [
            size
            for node, size in G.nodes(data='size')
            if G.nodes[node]['type'] == 'public_good'
        ]
        max_size = max(public_goods_sizes)
        min_size = min(public_goods_sizes)

        # Normalize function
        def normalize_size(size):
            if max_size == min_size:
                return 0.5
            return (size - min_size) / (max_size - min_size)

        # Updated color mapping function
        def get_node_color(node):
            if G.nodes[node]['type'] == 'voter':
                return get_hex_color(node)
            elif G.nodes[node]['type'] == 'public_good':
                size = G.nodes[node]['size']
                normalized = normalize_size(size)
                palette_length = len(
                    bokeh_RdYlGn[11]
                )  # Choose the palette length (11 is an example)
                return bokeh_RdYlGn[11][::-1][int(normalized * (palette_length - 1))]
            else:
                return custom_cmap.get(G.nodes[node].get('color', 'default'), 'default')

        tooltips = [
            ('Id', '@id'),
            ('Total Donations', '$@size{0,0.00}'),
            ('Type', '@type'),
        ]
        hover = HoverTool(tooltips=tooltips)

        # Create a dictionary to store the color of each node
        node_colors = {node: get_node_color(node) for node in G.nodes()}

        # Assign edge colors based on the color of the source (or voter) node
        for u, v, d in G.edges(data=True):
            d['color'] = node_colors[
                v
            ]  # or node_colors[v] depending on your preference

        # Visualization
        plot = hvnx.draw(
            G,
            pos=nx.spring_layout(G, seed=69),
            node_size='size',
            node_shape='shape',
            node_color=[get_node_color(node) for node in G.nodes()],
            edge_width='amountUSD',
            node_label='index',
            node_line_color='outline_color',
            node_line_width=2,
            edge_color='color',
            edge_alpha=0.8,
            node_alpha=0.95,
            cmap='viridis',
            width=800,
            height=800,
            title='Contributors and Public Goods Network',
        )

        # Adjust the points_df to have dummy x and y values
        points_df = pd.DataFrame(
            {
                'x': [0] * len(public_goods_sizes),
                'y': [0] * len(public_goods_sizes),
                'size': public_goods_sizes,
            }
        )

        # Create a Points plot for colorbar
        points_for_colorbar = hv.Points(points_df, kdims=['x', 'y']).opts(
            color='size',
            cmap=RdYlGn,
            colorbar=True,
            width=100,  # Narrow width for colorbar
            height=800,
            show_frame=False,
            xaxis=None,
            yaxis=None,
            toolbar=None,
            show_legend=False,
        )

        # Create the main graph plot without a colorbar
        main_graph_plot = plot.opts(
            hv.opts.Graph(
                padding=0.1,
                colorbar=False,
                legend_position='right',
                tools=[hover, 'tap'],
            ),
            hv.opts.Nodes(line_color='outline_color', line_width=5, tools=[hover]),
        )

        # Combine the plots into a layout
        layout = main_graph_plot + points_for_colorbar

        return layout

        def donation_groups_view(self):
            donations_df = self.donations.dataset
            plot = donations_df.groupby(['Grant Name'])
            return plot

    @pm.depends('donations.dataset')
    def view(self):
        return pn.Column(
            self,
            pn.Tabs(
                ('Projects', self.projects_view),
                ('Contributors', self.contributors_view),
                ('Contributions Network', self.contributions_network_view),
                # ('Contributions Matrix', self.contributions_matrix_view),
                # ('Donor Donation Counts', self.donor_view),
                # ('Sankey', self.sankey_view),
                active=2,
                dynamic=True,
            ),
        )
