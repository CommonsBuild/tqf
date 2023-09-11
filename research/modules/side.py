"""
This is a sideshow file. Code that doesn't quite make the cut for the main package.
"""


class Function(pm.Parameterized):
    """
    The identity function.
    The function base class and interface.
    """

    def f(self, x):
        return x


identity_function = Function()


class Sigmoid(Function):
    """
    This is a parameterized version of the sigmoid function.

    The sigmoid function is an S-shaped curve that can take any real-valued number
    and map it into a value between 0 and 1, but never exactly at those limits.
    It is widely used in machine learning, especially in the context of neural networks
    for the activation function.

    The general formula for the sigmoid function is:
    f(x) = A / (1 + exp(-k * (x - b)))

    """

    A = pm.Number(1, bounds=(0, 5), step=1, doc='Maximum value of the curve.')
    k = pm.Number(1, bounds=(0, 5), step=1, doc='Steepness of the curve.')
    b = pm.Number(0, bounds=(0, 5), step=1, doc='Shift of the curve.')

    def f(self, x):
        """
        The general formula for the sigmoid function is:
        f(x) = A / (1 + exp(-k * (x - b)))

        Returns
        -------
        float or array-like
            The computed sigmoid value(s) for the given input value(s).

        """
        return self.A / (1 + np.exp(-self.k * (x - self.b)))


sigmoid = Sigmoid()


class Transformer(pm.Parameterized):
    """
    Transformer object utilizes a function object.
    Transformer adds domain and applies the function.
    Transformers provide a view into a function.
    """

    function = pm.Selector(default=sigmoid, objects=[sigmoid, identity_function])

    def x(self):
        return np.linspace(0, 1, 1000)

    @pm.depends('function', watch=True)
    def f(self, x):
        return self.function.f(x)

    def df(self):
        xs = self.x()
        ys = self.f(xs)
        df = pd.DataFrame({'x': xs, 'y': ys}, columns=['x', 'y'])
        return df

    def chart_view(self):
        df = self.df()
        chart_view = df.hvplot.line(x='x', y='y', ylim=(-0.01, 1.01))
        return chart_view

    @pm.depends('function')
    def view(self):
        return pn.Row(self, self.chart_view)


transformer = Transformer(function=sigmoid)


class Boost(Transformer):
    token_distribution = pm.Selector()
