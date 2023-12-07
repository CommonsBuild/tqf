import panel as pn
import param as pm


class ItemList(pm.Parameterized):
    items = pm.ListSelector()

    def view_items(self):
        return pn.Column(*[item.view() for item in self.items])

    def view(self):
        return pn.Column(self, self.view_items())
