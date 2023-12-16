import numpy as np
import panel as pn
from bokeh.layouts import column
from bokeh.models import Button, ColumnDataSource, Ellipse, PointDrawTool
from bokeh.plotting import figure


# Function to calculate the bounding box of the points
def calculate_ellipse_bounds(x, y):
    center_x = np.mean(x)
    center_y = np.mean(y)
    width = max(x) - min(x)
    height = max(y) - min(y)
    return center_x, center_y, width, height


# Initial data
initial_data = dict(x=[1, 2, 3, 4, 5], y=[3, 6, 1, 5, 4])
source = ColumnDataSource(data=initial_data)

# Create a plot
plot = figure(x_range=(-10, 10), y_range=(-10, 10), title='Drag the points')
renderer = plot.circle('x', 'y', source=source, size=20, color='green', alpha=0.5)

# Add ellipse to the plot
center_x, center_y, width, height = calculate_ellipse_bounds(
    initial_data['x'], initial_data['y']
)
ellipse_glyph = Ellipse(
    x=center_x,
    y=center_y,
    width=width,
    height=height,
    fill_alpha=0.2,
    fill_color='blue',
)
ellipse_renderer = plot.add_glyph(ellipse_glyph)

# Callback to update the ellipse
def update_ellipse(attr, old, new):
    center_x, center_y, width, height = calculate_ellipse_bounds(
        source.data['x'], source.data['y']
    )
    ellipse_renderer.glyph.update(x=center_x, y=center_y, width=width, height=height)


source.on_change('data', update_ellipse)

# Add the PointDrawTool
draw_tool = PointDrawTool(renderers=[renderer], empty_value='black')
plot.add_tools(draw_tool)
plot.toolbar.active_tap = draw_tool

# Reset button
reset_button = Button(label='Reset')

# Callback for the reset button
def reset_data():
    source.data = initial_data


reset_button.on_click(reset_data)

# Create a Panel layout
layout = pn.Column(plot, reset_button)

# Serve the app
layout.servable()
