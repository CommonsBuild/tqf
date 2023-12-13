import panel as pn
import param

pn.extension()


class RiskComponent(pn.viewable.Viewer):
    portfolio = param.String(allow_refs=True)
    aggregation = param.Selector(default='month', objects=['month', 'year'])

    def __panel__(self):
        return pn.Column(
            '## Risk Component',
            self.param.aggregation,
            self._get_risk,
            styles={'border': '1px solid pink'},
        )

    @pn.depends('portfolio', 'aggregation')
    def _get_risk(self):
        return f'{self.portfolio}, {self.aggregation}'


class Application(pn.viewable.Viewer):
    portfolio = param.Selector(default='power', objects=['power', 'gas', 'co2'])

    def __panel__(self):
        return pn.Column(
            '# Application',
            self.param.portfolio,
            pn.Row(
                RiskComponent(portfolio=self.param.portfolio),
                RiskComponent(portfolio=self.param.portfolio),
            ),
            pn.Row(
                RiskComponent(portfolio=self.param.portfolio),
                RiskComponent(portfolio=self.param.portfolio),
            ),
            styles={'border': '2px solid black'},
        )


Application().servable()
