import panel as pn
import param as pm


# Assuming you have a Parameterized class like this
class A(pm.Parameterized):
    name = pm.String()
    value = pm.Number()


# Example list of items
items = [A(name='Item 1', value=1), A(name='Item 2', value=2)]


class CRUDWidget(pm.Parameterized):
    selected_item = pm.Selector(objects=items)
    edit_mode = pm.Boolean(default=False)

    def __init__(self, **params):
        super().__init__(**params)
        self.list_view = pn.Column(*[self.create_item_view(item) for item in items])
        self.detail_view = pn.Param(
            self.selected_item,
            widgets={
                'name': {'type': pn.widgets.TextInput},
                'value': {'type': pn.widgets.FloatInput},
            },
        )

    def create_item_view(self, item):
        return pn.Row(
            pn.pane.Str(item.name),
            pn.widgets.Button(name='🖊️', button_type='primary', width=50, align='end'),
            pn.widgets.Button(name='❌', button_type='danger', width=50, align='end'),
        )

    def view(self):
        return pn.Column(
            pn.Row(
                pn.widgets.Button(name='➕', button_type='success', width=50),
                align='start',
            ),
            self.list_view,
            self.detail_view,
        )


crud_widget = CRUDWidget()
crud_widget.view().servable()
