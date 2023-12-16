import numpy as np
import panel as pn
import param as pm
from bokeh.models import PointDrawTool
from bokeh.plotting import figure

pn.extension()

# Shared figure for all shapes
shared_figure = figure(x_range=(-3, 3), y_range=(-3, 3))

# Add the PointDrawTool to the shared figure for all renderers
# draw_tool = PointDrawTool()
# shared_figure.add_tools(draw_tool)
# shared_figure.toolbar.active_tap = draw_tool


class Point(pm.Parameterized):
    x = pm.Number(default=0)
    y = pm.Number(default=0)


class Shape(pm.Parameterized):
    radius = pm.Number(default=1, bounds=(0, 1))
    center = pm.Parameter(default=Point())

    def __init__(self, **params):
        self.renderer = shared_figure.line(*self._get_coords())
        super(Shape, self).__init__(**params)
        # self.add_point_draw_tool()

    # def add_point_draw_tool(self):
    #     draw_tool = PointDrawTool(renderers=[self.renderer])
    #     shared_figure.add_tools(draw_tool)
    #     shared_figure.toolbar.active_tap = draw_tool

    def _get_coords(self):
        return [], []

    def view(self):
        return shared_figure

    def translate_coords(self, xs, ys):
        return xs + self.center.x, ys + self.center.y


class Circle(Shape):
    n = pm.Integer(default=100, precedence=-1)

    def _get_coords(self):
        angles = np.linspace(0, 2 * np.pi, self.n + 1)
        xs = self.radius * np.sin(angles)
        ys = self.radius * np.cos(angles)
        return self.translate_coords(xs, ys)

    @pm.depends('radius', 'center.x', 'center.y', watch=True, on_init=True)
    def update(self):
        xs, ys = self._get_coords()
        self.renderer.data_source.data.update({'x': xs, 'y': ys})


class NGon(Circle):
    n = pm.Integer(default=3, bounds=(3, 10), precedence=1)

    @pm.depends('radius', 'n', 'center.x', 'center.y', watch=True, on_init=True)
    def update(self):
        xs, ys = self._get_coords()
        self.renderer.data_source.data.update({'x': xs, 'y': ys})


# Instantiate and update shapes with different centers
shapes = [NGon(center=Point(x=1, y=1)), Circle(center=Point(x=-1, y=-1))]


class ShapeViewer(pm.Parameterized):
    shape = pm.ObjectSelector(default=shapes[0], objects=shapes)

    @pm.depends('shape')
    def view(self):
        return self.shape.view()

    @pm.depends('shape', 'shape.radius', 'shape.center.x', 'shape.center.y')
    def title(self):
        return '## %s (radius=%.1f, center=(%.1f, %.1f))' % (
            type(self.shape).__name__,
            self.shape.radius,
            self.shape.center.x,
            self.shape.center.y,
        )

    def panel(self):
        return pn.Column(self.title, self.view)


shape_viewer = ShapeViewer()

layout = pn.Row(shape_viewer.param, shape_viewer.panel())

layout.servable()
