import holoviews as hv
import hvplot.networkx as hvnx
import networkx as nx
import numpy as np
import pandas as pd
import panel as pn
import param as pm
from bokeh.models import HoverTool

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
        df = self.donations.dataset
        donor_vote_counts = (
            df.groupby('voter').count()['id'].to_frame(name='number_of_donations')
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
        df = self.donations.dataset
        sankey = hv.Sankey(df[['voter', 'Grant Name', 'amountUSD']])
        return sankey

    @pm.depends('donations.dataset')
    def projects_view(self):
        df = self.donations.dataset

        # Calculate Data per Project
        projects = (
            df.groupby('Grant Name')
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
        df = self.donations.dataset

        # Calculate Data per Contributor
        contributors = (
            df.groupby('voter')
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
        df = self.donations.dataset
        contributions_matrix = df.pivot_table(
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
        df = self.donations.dataset.replace(0, np.nan)

        df['voter'] = df['voter'].astype(str)
        df['Grant Name'] = df['Grant Name'].astype(str)

        # Create graph from the dataframe
        G = nx.from_pandas_edgelist(
            df,
            'voter',
            'Grant Name',
            ['amountUSD'],
            create_using=nx.Graph(),
        )

        # Modify edge width to be the donation size divided by 10
        for u, v, d in G.edges(data=True):
            d['amountUSD'] = d['amountUSD'] / 40

        # Set node attributes
        for node in G.nodes():
            if node in df['voter'].unique():
                G.nodes[node]['size'] = df[df['voter'] == node]['amountUSD'].sum()
                G.nodes[node]['id'] = node
                G.nodes[node]['shape'] = 'circle'
                G.nodes[node]['type'] = 'voter'
                G.nodes[node]['outline_color'] = 'blue'  # Outline color for voters
            else:
                G.nodes[node]['size'] = df[df['Grant Name'] == node]['amountUSD'].sum()
                G.nodes[node]['id'] = node
                G.nodes[node]['shape'] = 'triangle'
                G.nodes[node]['type'] = 'public_good'
                G.nodes[node]['outline_color'] = 'red'  # Outline color for voters

        tooltips = [
            ('Id', '@id'),
            ('Total Donations', '$@size{0,0.00}'),
            ('Type', '@type'),
        ]
        hover = HoverTool(tooltips=tooltips)

        # Visualization
        plot = hvnx.draw(
            G,
            pos=nx.spring_layout(G, seed=69),
            node_size='size',
            node_shape='shape',
            node_color='size',
            edge_width='amountUSD',
            node_label='index',
            node_line_color='outline_color',
            node_line_width=2,
            edge_color='amountUSD',
            edge_alpha=0.8,
            node_alpha=0.95,
            cmap='viridis',
            width=800,
            height=800,
            title='Contributors and Public Goods Network',
        )

        plot.opts(
            hv.opts.Graph(
                padding=0.1,
                colorbar=True,
                legend_position='right',
                tools=[hover, 'tap'],
            ),
            hv.opts.Nodes(line_color='outline_color', line_width=5, tools=[hover]),
        )
        return plot

    def donation_groups_view(self):
        df = self.donations.dataset
        plot = df.groupby(['Grant Name'])
        return plot

    @pm.depends('donations.dataset')
    def view(self):
        return pn.Column(
            self,
            pn.Tabs(
                ('Projects', self.projects_view),
                ('Contributors', self.contributors_view),
                ('Contributions Matrix', self.contributions_matrix_view),
                ('Contributions Network', self.contributions_network_view),
                # ('Donor Donation Counts', self.donor_view),
                # ('Sankey', self.sankey_view),
                active=3,
                dynamic=True,
            ),
        )
