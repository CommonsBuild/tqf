import holoviews as hv
import networkx as nx
import numpy as np
import pandas as pd
from bokeh.models import HoverTool
from holoviews import opts

hv.extension('bokeh')

# Sample data
data = {
    'Grant Name': ['1', 'A', 'B', 'C', 'D', 'E'],
    'Total Donations': [5, 275, 290, 300, 500, 450],
}
public_goods_data = pd.DataFrame(data)

# Add 'x' and 'y' columns for scatter plot
public_goods_data['x'] = np.zeros(len(public_goods_data))  # Example: [0, 1, 2]
public_goods_data['y'] = public_goods_data['Total Donations']

# Scatter plot with colorbar
points = hv.Points(
    public_goods_data, kdims=['x', 'y'], vdims=['Grant Name', 'Total Donations']
).opts(
    size=7,
    marker='square',
    color='y',
    line_color='black',
    cmap='RdYlGn',  # Use a colormap name
    colorbar=True,
    width=200,
    height=800,
    xaxis=None,
    toolbar=None,
    show_frame=False,
    tools=[
        HoverTool(
            tooltips=[
                ('Grant Name', '@{Grant Name}'),
                ('Total Donations', '@{Total Donations}'),
            ]
        )
    ],
)

# Display
hv.save(points, 'example.html')
