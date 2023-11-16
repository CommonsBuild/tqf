import colorcet as cc
import holoviews as hv
import hvplot.networkx as hvnx
import networkx as nx
import numpy as np
import pandas as pd
import panel as pn
import param as pm
from bokeh.models import (
    AdaptiveTicker,
    BasicTicker,
    ColorBar,
    CustomJSTickFormatter,
    FixedTicker,
    HoverTool,
    LinearColorMapper,
    LogTicker,
    PrintfTickFormatter,
)
from bokeh.palettes import RdYlGn as bokeh_RdYlGn

RdYlGn = bokeh_RdYlGn[10][::-1]  # This reverses the chosen palette

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

    def contributor_dataset(self):
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

        return contributors

    @pm.depends('donations.dataset')
    def contributors_view(self):

        contributors = self.contributor_dataset()

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
                G.nodes[node]['shape'] = 'square'
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

        # Graph Layout Position
        pos = nx.spring_layout(G, seed=69)

        # Visualization
        plot = hvnx.draw(
            G,
            pos=pos,
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
        ).opts(hv.opts.Graph(title='Public Goods Contributions Network'))

        # Create the main graph plot without a colorbar
        main_graph_plot = plot.opts(
            hv.opts.Graph(
                padding=0.01,
                colorbar=False,
                legend_position='right',
                tools=[hover, 'tap'],
            ),
            hv.opts.Nodes(line_color='outline_color', line_width=5, tools=[hover]),
        )

        # Adjust the code to create a DataFrame for labels
        label_data = []
        for node, data in G.nodes(data=True):
            if data.get('type') == 'public_good':
                pos = nx.spring_layout(G, seed=69)[
                    node
                ]  # Adjust the position calculation as needed
                label_data.append({'x': pos[0], 'y': pos[1], 'label': data['id']})

        # Convert to DataFrame
        label_df = pd.DataFrame(label_data)

        # Adjust label positions (e.g., move them to the right)
        label_df['x'] += 0  # Adjust this value as needed

        # Create labels using the DataFrame
        labels = hv.Labels(label_df, kdims=['x', 'y'], vdims='label').opts(
            text_font_size='10pt'
        )

        # Overlay labels on the network graph
        main_graph_plot = main_graph_plot * labels

        # Create a DataFrame for the Points plot with 'Grant Name' and 'Total Donations'
        public_goods_data = (
            donations_df.groupby('Grant Name')['amountUSD'].sum().reset_index()
        )
        public_goods_data.rename(columns={'amountUSD': 'Total Donations'}, inplace=True)

        def calculate_size(donation_amount):
            # Example logic for calculating size
            # This can be adjusted based on how you want to scale the sizes
            base_size = 8  # Base size for the points
            scaling_factor = 0.02  # Scaling factor for donation amounts
            return base_size + (donation_amount * scaling_factor)

        # All points have the same x-coordinate, y is the actual 'Total Donations'
        public_goods_data['x'] = 0
        public_goods_data['y'] = public_goods_data['Total Donations']
        public_goods_data['size'] = public_goods_data['Total Donations'].apply(
            lambda x: calculate_size(x)
        )  # Define calculate_size based on your graph's logic

        # Assuming public_goods_data is your DataFrame
        max_funding = public_goods_data['Total Donations'].max()
        high_value = max_funding * 1.05  # 110% of the max funding

        # Create a color mapper with specified range
        color_mapper = LinearColorMapper(palette=RdYlGn, low=0, high=high_value)

        # Create a Points plot for the colorbar and hover information
        public_goods_data = public_goods_data.rename(
            columns={'Grant Name': 'grant_name', 'Total Donations': 'total_donations'}
        )
        points_for_colorbar = hv.Points(
            public_goods_data,
            kdims=['x', 'y'],
            vdims=['grant_name', 'total_donations', 'size'],
        ).opts(
            size='size',
            marker='square',
            color='y',
            line_color='black',
            line_width=2,
            cmap=color_mapper,  # Assuming RdYlGn is your colormap variable
            colorbar=True,
            width=300,
            height=800,
            show_frame=False,
            xaxis=None,
            yaxis=None,
            toolbar=None,
            show_legend=False,
            title='Public Goods Funding Outcomes',
            ylim=(
                -100,
                high_value,
            ),  # Assuming 'high_value' is your calculated upper limit
            tools=[
                HoverTool(
                    tooltips=[
                        ('Grant Name', '@grant_name'),
                        ('Total Donations', '$@total_donations{0,0.00}'),
                    ]
                )
            ],
        )

        # Adjust colorbar with the fixed range
        points_for_colorbar = points_for_colorbar.opts(
            hv.opts.Points(
                colorbar=True,
                cmap=RdYlGn,
                colorbar_opts={
                    # 'color_mapper': color_mapper,
                    'ticker': AdaptiveTicker(desired_num_ticks=10),
                    'formatter': PrintfTickFormatter(format='%d'),
                },
            )
        )

        # Define a fixed offset for scatter point labels
        scatter_label_offset_x = 0  # Adjust this value as needed
        scatter_label_offset_y = 0.0   # Y offset, if needed

        # Create a DataFrame for labels with positions adjusted
        label_data = public_goods_data.copy()
        label_data['x'] = label_data['x'] + scatter_label_offset_x
        label_data['y'] = label_data['y'] + scatter_label_offset_y

        # Create labels for scatter points
        scatter_labels = hv.Labels(
            label_data, kdims=['x', 'y'], vdims=['grant_name']
        ).opts(text_font_size='10pt')

        # Adjust width of the scatter plot to accommodate labels and colorbar
        plot_width = 400  # Adjust this value as needed to fit labels

        # Update the scatter plot with new width and overlay labels
        points_for_colorbar = (
            points_for_colorbar.opts(width=plot_width) * scatter_labels
        )

        # bars_with_colorbar = public_goods_data.hvplot.bar(
        #     y='total_donations',
        #     x='grant_name',
        #     color='total_donations',
        #     cmap=RdYlGn,
        #     width=300,
        #     height=1000,
        #     sort='total_donations',
        #     rot=90,
        # )

        # Combine the plots into a layout
        layout = (
            main_graph_plot.opts(shared_axes=False)
            + points_for_colorbar.opts(shared_axes=False)
        ).opts(shared_axes=False)

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
                ('Contributions Matrix', self.contributions_matrix_view),
                # ('Donor Donation Counts', self.donor_view),
                # ('Sankey', self.sankey_view),
                active=2,
                dynamic=True,
            ),
        )
