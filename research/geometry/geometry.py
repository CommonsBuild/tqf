import numpy as np
import panel as pn
import param
from bokeh.plotting import figure

pn.extension()

# Shared figure for all shapes
shared_figure = figure(x_range=(-3, 3), y_range=(-3, 3))


class Point(param.Parameterized):
    x = param.Number(default=0)
    y = param.Number(default=0)


class Shape(param.Parameterized):
    radius = param.Number(default=1, bounds=(0, 1))
    center = param.Parameter(default=Point())

    def __init__(self, **params):
        self.renderer = shared_figure.line(*self._get_coords())
        super(Shape, self).__init__(**params)

    def _get_coords(self):
        return [], []

    def view(self):
        return shared_figure

    def translate_coords(self, xs, ys):
        return xs + self.center.x, ys + self.center.y


class Circle(Shape):
    n = param.Integer(default=100, precedence=-1)

    def _get_coords(self):
        angles = np.linspace(0, 2 * np.pi, self.n + 1)
        xs = self.radius * np.sin(angles)
        ys = self.radius * np.cos(angles)
        return self.translate_coords(xs, ys)

    @param.depends('radius', 'center.x', 'center.y', watch=True, on_init=True)
    def update(self):
        xs, ys = self._get_coords()
        self.renderer.data_source.data.update({'x': xs, 'y': ys})


class NGon(Circle):
    n = param.Integer(default=3, bounds=(3, 10), precedence=1)

    @param.depends('radius', 'n', 'center.x', 'center.y', watch=True, on_init=True)
    def update(self):
        xs, ys = self._get_coords()
        self.renderer.data_source.data.update({'x': xs, 'y': ys})


# Instantiate and update shapes with different centers
shapes = [NGon(center=Point(x=1, y=1)), Circle(center=Point(x=-1, y=-1))]


class ShapeViewer(param.Parameterized):
    shape = param.ObjectSelector(default=shapes[0], objects=shapes)

    @param.depends('shape')
    def view(self):
        return self.shape.view()

    @param.depends('shape', 'shape.radius', 'shape.center.x', 'shape.center.y')
    def title(self):
        return '## %s (radius=%.1f, center=(%.1f, %.1f))' % (
            type(self.shape).__name__,
            self.shape.radius,
            self.shape.center.x,
            self.shape.center.y,
        )

    def panel(self):
        return pn.Column(self.title, self.view)


viewer = ShapeViewer()

pn.Row(viewer.param, viewer.panel()).servable()
