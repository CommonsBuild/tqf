from math import log

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
    LogColorMapper,
    LogTicker,
    PrintfTickFormatter,
    TickFormatter,
)
from bokeh.palettes import Greens
from bokeh.palettes import RdYlGn as bokeh_RdYlGn
from bokeh.transform import log_cmap

RdYlGn = bokeh_RdYlGn[11][::-1]  # This reverses the chosen palette
Greens = Greens[256][::-1]

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
    def projects_table(self, donations_df, donation_column='amountUSD'):

        total_donations = donations_df[donation_column].sum()
        total_donors = donations_df['voter'].nunique()

        # Calculate Data per Project
        projects = (
            donations_df.groupby('Grant Name')
            .apply(
                lambda group: pd.Series(
                    {
                        'Number of Donors': group['voter'].nunique(),
                        'Percentage of Donors': group['voter'].nunique() / total_donors,
                        'Total Donations': group[donation_column].sum(),
                        'Percent of Donations': group[donation_column].sum()
                        / total_donations,
                        'Mean Donation': group[donation_column].mean(),
                        'Median Donation': group[donation_column].median(),
                        'Max Donations': group[donation_column].max(),
                        'Max Donor': group.loc[
                            group[donation_column].idxmax(), 'voter'
                        ],
                        'Donations': sorted(
                            group[donation_column].tolist(), reverse=True
                        ),
                    }
                )
            )
            .reset_index()
        )

        return projects

    @pm.depends('donations.dataset')
    def projects_view(self):
        projects = self.projects_table(donations_df=self.donations.dataset)

        # Format the donations list
        projects['Donations'] = projects['Donations'].apply(
            lambda donations: ['${:.2f}'.format(n) for n in donations]
        )

        # Use tabulator to display the data
        projects_view = pn.widgets.Tabulator(
            projects,
            formatters={'Donations': {'type': 'textarea', 'textAlign': 'left'}},
            pagination=None,
        )
        projects_view.style.map(color_based_on_eth_address, subset='Max Donor')
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
        contributors_view.style.map(color_based_on_eth_address, subset='voter')
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
        numeric_columns = contributions_matrix.select_dtypes(include=['number']).columns
        contributions_matrix_view = pn.widgets.Tabulator(
            contributions_matrix,
            widths={col: 120 for col in numeric_columns},
            aggregators={col: 'sum' for col in numeric_columns},
            pagination=None,
        )
        contributions_matrix_view.style.map(
            color_based_on_eth_address,
            subset='voter',
        )
        # Set the minimum and maximum values for the colormap
        min_value = contributions_matrix[numeric_columns].replace(0, np.nan).min().min()
        max_value = contributions_matrix[numeric_columns].max().max()

        # Create the logarithmic color mapper
        log_mapper = log_cmap('value', palette=Greens, low=min_value, high=max_value)

        # Define the coloring function with text color adjustment
        def color_cell(cell_value):
            if (cell_value <= 0) or np.isnan(cell_value):
                return 'background-color: white; color: black;'
            # Normalize the logarithmic value and map it to the reversed Greens palette
            normalized_value = (log(cell_value) - log(min_value)) / (
                log(max_value) - log(min_value)
            )
            color_index = int(normalized_value * (len(Greens) - 1))

            # Determine text color based on background darkness
            text_color = 'white' if color_index > len(Greens) // 2 else 'black'

            return f'background-color: {Greens[color_index]}; color: {text_color};'

        for col in numeric_columns:
            contributions_matrix_view.style.map(color_cell, subset=[col])

        return contributions_matrix_view

    @pm.depends('donations.dataset')
    def contributions_matrix_heatmap_view(self):
        contributions_matrix = (
            self.contributions_matrix()
            .reset_index()
            .replace(np.nan, 0)
            .set_index('voter')
        )

        heatmap = (
            (contributions_matrix + 1)
            .hvplot.heatmap(
                title='Contributions Matrix',
                cmap='Greens',
                clim=(1, contributions_matrix.max().max()),
                fontscale=1.2,
                width=80 * contributions_matrix.columns.nunique(),
                height=20 * contributions_matrix.index.nunique(),
                xlabel='Public Good',
                ylabel='Citizen',
                clabel='Amount of value produced by public_good p for citizen i.',
                cnorm='log',
                rot=90,
                xaxis='top',
                yaxis='right',
            )
            .opts(default_tools=['pan', 'hover'])
        )

        # Customize the plot using HoloViews and Bokeh
        def apply_custom_styling(plot, element):
            plot.state.axis.major_label_text_color = (
                'red'  # Example to change label text color
            )
            # Add more customizations here

        # Apply the custom styling
        heatmap = heatmap.opts(hv.opts.HeatMap(hooks=[apply_custom_styling]))

        return pn.Column(heatmap, contributions_matrix)

    @pm.depends('donations.dataset')
    def contributions_network_view(self):
        return self._contributions_network_view(self.donations.dataset)

    def _contributions_network_view(
        self, donations_df, donations_column='amountUSD', funding_outcomes=None
    ):
        # Replace nan values with 0
        donations_df = donations_df.replace(0, np.nan)

        # Explicitly set string columns as str
        donations_df['voter'] = donations_df['voter'].astype(str)
        donations_df['Grant Name'] = donations_df['Grant Name'].astype(str)

        # Create graph from the dataframe
        G = nx.from_pandas_edgelist(
            donations_df,
            'voter',
            'Grant Name',
            [donations_column],
            create_using=nx.DiGraph(),
        )

        # Modify edge width to be the donation size divided by 10
        for u, v, d in G.edges(data=True):
            # d['edge_width'] = np.sqrt(d[donations_column]) / 5
            # d['weight'] = np.sqrt(d[donations_column])
            d['edge_width'] = d[donations_column] / 30
            d['weight'] = d[donations_column]

        # Assigning custom colors per node type
        # voter_color_value = 1
        # public_good_color_value = 2
        # custom_cmap = {
        #     voter_color_value: 'purple',
        #     public_good_color_value: 'orange',
        #     'default': 'lightgray',
        # }

        # Funding Outcomes
        if funding_outcomes is None:
            total_funding_per_public_good = donations_df.groupby('Grant Name')[
                donations_column
            ].sum()

        else:
            total_funding_per_public_good = funding_outcomes['Matching Funds Boosted']

        # Find the max and min total donations for normalization
        max_size = total_funding_per_public_good.max()
        min_size = total_funding_per_public_good.min()

        def get_hex_color(address):
            hex_color = f'#{address[2:8]}'
            return hex_color

        def get_donor_edge_color(address):
            if not hasattr(self, 'boost_tables'):
                return get_hex_color(address)
            else:
                return 'black'

        # Set node attributes
        for node in G.nodes():
            if node in donations_df['voter'].unique():
                G.nodes[node]['size'] = donations_df[donations_df['voter'] == node][
                    donations_column
                ].sum()
                # G.nodes[node]['color'] = voter_color_value
                G.nodes[node]['id'] = node
                G.nodes[node]['shape'] = 'circle'
                G.nodes[node]['type'] = 'voter'
                G.nodes[node]['outline_color'] = get_donor_edge_color(node)
            else:
                if funding_outcomes is None:
                    G.nodes[node]['size'] = donations_df[
                        donations_df['Grant Name'] == node
                    ][donations_column].sum()
                else:
                    G.nodes[node]['size'] = funding_outcomes[
                        funding_outcomes['Grant Name'] == node
                    ]['Matching Funds Boosted'].iloc[0]
                G.nodes[node]['id'] = node
                G.nodes[node]['shape'] = 'square'
                G.nodes[node]['type'] = 'public_good'
                G.nodes[node]['outline_color'] = 'black'  # Outline color for voters
                # G.nodes[node]['color'] = public_good_color_value

        # Now, calculate max_size and min_size based on the node attributes
        # public_goods_sizes = np.log(
        #     [
        #         size
        #         for node, size in G.nodes(data='size')
        #         if G.nodes[node]['type'] == 'public_good'
        #     ]
        # )
        # max_size = max(public_goods_sizes)
        # min_size = min(public_goods_sizes)
        #
        # # Normalize function
        # def normalize_size(size):
        #     if max_size == min_size:
        #         return 0.5
        #     return (np.log(size) - min_size) / (max_size - min_size)

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
                if hasattr(self, 'sme_list') and node in self.sme_list:
                    return get_hex_color(node)
                else:
                    return 'white'
            elif G.nodes[node]['type'] == 'public_good':
                size = G.nodes[node]['size']
                normalized = normalize_size(size)
                palette_length = len(
                    bokeh_RdYlGn[11]
                )  # Choose the palette length (11 is an example)
                return bokeh_RdYlGn[11][::-1][int(normalized * (palette_length - 1))]
            else:
                return custom_cmap.get(G.nodes[node].get('color', 'default'), 'default')

        # Updated color mapping function
        def get_node_alpha(node):
            if G.nodes[node]['type'] == 'voter':
                if hasattr(self, 'sme_list') and node in self.sme_list:
                    return 0.95
                else:
                    return 0.95
            elif G.nodes[node]['type'] == 'public_good':
                return 0.95

        def get_edge_alpha(edge):
            if hasattr(self, 'sme_list') and edge[0] in self.sme_list:
                # return 0.95
                return 0
            else:
                return 0

        tooltips = [
            ('Id', '@id'),
            ('Matching Funds Boosted', '$@size{0,0.00}'),
            ('Type', '@type'),
        ]
        hover = HoverTool(tooltips=tooltips)

        # Create a dictionary to store the color of each node
        node_colors = {node: get_node_color(node) for node in G.nodes()}

        # Assign edge colors based on the color of the source (or voter) node
        for u, v, d in G.edges(data=True):
            d['color'] = node_colors[
                u
            ]  # or node_colors[v] depending on your preference

        # Create a DataFrame for the Points plot with 'Grant Name' and 'Matching Funds Boosted'
        if funding_outcomes is None:
            public_goods_data = (
                donations_df.groupby('Grant Name')[donations_column].sum().reset_index()
            )
            public_goods_data.rename(
                columns={donations_column: 'Matching Funds Boosted'}, inplace=True
            )
        else:
            public_goods_data = funding_outcomes

        def calculate_size(donation_amount):
            # Example logic for calculating size
            # This can be adjusted based on how you want to scale the sizes
            base_size = 8  # Base size for the points
            scaling_factor = 0.02  # Scaling factor for donation amounts
            return base_size + (donation_amount * scaling_factor)

        # All points have the same x-coordinate, y is the actual 'Matching Funds Boosted'
        public_goods_data['x'] = 0
        public_goods_data['y'] = public_goods_data['Matching Funds Boosted']
        public_goods_data['size'] = public_goods_data['Matching Funds Boosted'].apply(
            lambda x: calculate_size(x)
        )  # Define calculate_size based on your graph's logic

        # Assuming public_goods_data is your DataFrame
        max_funding = public_goods_data['Matching Funds Boosted'].max()
        min_funding = public_goods_data['Matching Funds Boosted'].min()
        # high_value = max_funding * 1.3  # 110% of the max funding
        # low_value = min_funding * 0.90  # 110% of the max funding
        high_value = max_funding * 1.05  # 110% of the max funding
        low_value = 0  # 110% of the max funding

        # Create a color mapper with specified range
        # color_mapper = LogColorMapper(palette=RdYlGn, low=1, high=high_value)
        color_mapper = LinearColorMapper(palette=RdYlGn, low=0, high=high_value)

        # Create a Points plot for the colorbar and hover information
        public_goods_data[['grant_name', 'total_funding']] = public_goods_data[
            ['Grant Name', 'Matching Funds Boosted']
        ]
        points_for_colorbar = hv.Points(
            public_goods_data,
            kdims=['x', 'y'],
            vdims=['grant_name', 'total_funding', 'size'],
        ).opts(
            size='size',
            marker='square',
            color='y',
            line_color='black',
            line_width=2,
            # cmap=color_mapper,  # Assuming RdYlGn is your colormap variable
            colorbar=True,
            # colorbar=False,
            cmap='RdYlGn',
            alpha=0,
            # padding=0.5,
            # logy=True,
            # logz=True,
            width=1000,
            height=800,
            # show_frame=False,
            # xaxis=None,
            # yaxis=None,
            # toolbar=None,
            show_legend=False,
            title='Public Goods Funding Outcomes',
            ylim=(
                low_value,
                high_value,
            ),  # Assuming 'high_value' is your calculated upper limit
            xlim=(-5, 5),
            tools=[
                HoverTool(
                    tooltips=[
                        ('Grant Name', '@grant_name'),
                        ('Matching Funds Boosted', '$@total_funding{0,0.00}'),
                    ]
                )
            ],
            colorbar_opts={
                # 'color_mapper': color_mapper,
                # 'ticker': FixedTicker(
                #     ticks=[int(x) for x in range(0, int(max_funding), 200)]
                # ),
                'formatter': PrintfTickFormatter(format='%d'),
            },
        )

        def customize_colorbar(plot, element):
            color_bar = plot.handles.get('color_bar', None)
            if color_bar:
                color_bar.major_label_text_font_size = (
                    '14pt'  # Adjust the font size as needed
                )

        points_for_colorbar = points_for_colorbar.opts(hooks=[customize_colorbar])

        # Define a fixed offset for scatter point labels
        # scatter_label_offset_x = 2.5  # Adjust this value as needed
        # scatter_label_offset_y = 0.0   # Y offset, if needed

        # Create a DataFrame for labels with positions adjusted
        # label_data = public_goods_data.copy()
        #
        # label_data['grant_name_padded'] = label_data['grant_name'].str.pad(
        #     width=label_data['grant_name'].str.len().max(),
        #     side='right',
        #     fillchar=' ',
        # )
        #
        # label_data['x'] = (
        #     label_data['x']
        #     + scatter_label_offset_x
        #     + 0.02 * label_data['grant_name'].str.len()
        # )
        # label_data['y'] = label_data['y'] + scatter_label_offset_y
        #
        # # Create labels for scatter points
        # scatter_labels = hv.Labels(
        #     label_data, kdims=['x', 'y'], vdims=['grant_name']
        # ).opts(text_font_size='10pt', text_align='right', text_baseline='middle')

        # Adjust width of the scatter plot to accommodate labels and colorbar
        plot_width = 1200  # Adjust this value as needed to fit labels

        # Update the scatter plot with new width and overlay labels
        points_for_colorbar = (
            points_for_colorbar.opts(
                width=plot_width,
                colorbar_position='left',
                fontsize={
                    'title': 22,
                    'ticks': 14,
                },
                # fontscale=2,
            )
            # * scatter_labels
        )

        # public_goods_positions = points_for_colorbar.Points.I.data[
        #     ['grant_name', 'x', 'y']
        # ]
        public_goods_positions = points_for_colorbar.data[['grant_name', 'x', 'y']]

        pos = {
            p['grant_name']: (p['x'], p['y'])
            for p in public_goods_positions.to_dict(orient='records')
        }
        fixed = list(pos.keys())

        # Graph Layout Position
        k = 10 / len(fixed)
        pos = nx.spring_layout(
            G, k=k, iterations=100, weight='edge_width', seed=69, fixed=fixed, pos=pos
        )
        # pos = nx.spring_layout(G, k=k, iterations=100, weight='edge_width', seed=69)
        # pos = nx.spring_layout(G, k=k, iterations=100, weight='edge_width', seed=69)

        # Reflect on y axis:
        pos = {n: (-abs(x), y) for n, (x, y) in pos.items()}

        pos = {n: (x + 1, y) if n in fixed else (x, y) for n, (x, y) in pos.items()}

        # Visualization
        plot = hvnx.draw(
            G,
            pos=pos,
            fixed=fixed,
            arrows=False,
            edge_curvature=0.5,
            node_size='size',
            node_shape='shape',
            node_color=[get_node_color(node) for node in G.nodes()],
            edge_width='edge_width',
            # node_label='index',
            node_line_color='outline_color',
            node_line_width=2,
            edge_color='color',
            edge_alpha=[get_edge_alpha(edge) for edge in G.edges(data=True)],
            node_alpha=[get_node_alpha(node) for node in G.nodes()],
            cmap='RdYlGn',
            width=1800,
            height=1000,
        ).opts(hv.opts.Graph(title='Public Goods Donations and Funding outcomes'))

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
                node_pos = pos[node]  # Adjust the position calculation as needed
                label_data.append(
                    {'x': node_pos[0], 'y': node_pos[1], 'label': data['id']}
                )

        # Convert to DataFrame
        label_df = pd.DataFrame(label_data)

        # Adjust label positions (e.g., move them to the right)
        label_df['x'] += 0  # Adjust this value as needed

        # Create labels using the DataFrame
        labels = hv.Labels(label_df, kdims=['x', 'y'], vdims='label').opts(
            text_font_size='11pt',
            text_align='left',
            xoffset=0.4,
        )

        # Overlay labels on the network graph
        main_graph_plot = main_graph_plot * labels

        layout = (
            main_graph_plot.opts(shared_axes=False)
            * points_for_colorbar.opts(shared_axes=False)
        ).opts(shared_axes=False)

        # return nx.to_pandas_edgelist(G)
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
                # ('Contributions Network', self.contributions_network_view),
                ('Contributions Matrix', self.contributions_matrix_view),
                # ('Donor Donation Counts', self.donor_view),
                # ('Sankey', self.sankey_view),
                active=0,
                dynamic=True,
            ),
        )
