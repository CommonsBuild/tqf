import panel as pn
from bokeh.layouts import column
from bokeh.models import Button, ColumnDataSource, PointDrawTool
from bokeh.plotting import curdoc, figure

# Create the plot
plot = figure(x_range=(-10, 10), y_range=(-10, 10), title='Drag the points')
# renderer = plot.circle('x', 'y', source=source, size=20, color='green', alpha=0.5)
initial_data = dict(
    x=[1, 2, 3, 4, 5],
    y=[3, 6, 1, 5, 4],
    radius=[0.3, 0.6, 0.1, 0.5, 0.4],
    alpha=[0.3, 0.6, 0.1, 0.5, 0.4],
)
# Prepare the data
source = ColumnDataSource(data=initial_data)
# renderer = plot.circle('x', 'y', 'radius', 'alpha', source=source, color='green')
renderer = plot.circle(
    'x', 'y', radius='radius', alpha='alpha', source=source, color='green'
)


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
