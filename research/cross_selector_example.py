import panel as pn
import param as pm


class A(pm.Parameterized):
    a = pm.Integer(default=0)


a = A()
b = A()


class MultiSelect(pm.Parameterized):
    # A ListSelector parameter, which will be represented by a CrossSelector in the UI
    options = pm.ListSelector(
        default=[],
        objects=[a, b],
    )


# Create an instance of your parameterized class
select = MultiSelect()

# Automatically create a UI for the parameters
param_panel = pn.Column(pn.panel(select.param, expand_button=True, expand=True))

# Display the panel
param_panel.servable()

# Display with controls
# pn.Row(param_panel.controls(jslink=True), param_panel).servable()
