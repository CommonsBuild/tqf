import panel as pn
import param as pm

from .boost import Boost


class BoostFactory(pm.Parameterized):
    template = pm.Selector()
    boosts = pm.List(default=[], class_=Boost, precedence=-1)
    new_boost = pm.Action(lambda self: self._new_boost())
    remove_boost = pm.Action(lambda self: self._remove_boost())

    def _new_boost(self):
        self.boosts.append(Boost(**self.template.param.values()))
        self.param.trigger('boosts')

    def _remove_boost(self):
        if len(self.boosts):
            self.boosts.pop()
            self.param.trigger('boosts')

    @pm.depends('boosts')
    def boosts_view(self):
        return pn.Column(*[boost.view for boost in self.boosts])

    @pm.depends('boosts')
    def view(self):
        return pn.Row(self, self.boosts_view)
