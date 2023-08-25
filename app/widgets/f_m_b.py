import hvplot.pandas
import panel as pn
import param
import numpy as np
import pandas as pd
import holoviews as hv
pn.extension('mathjax')

class QuadraticFunding(param.Parameterized):
    F = param.Number(default=1_000, bounds=(0, 10_000), step=100, label="Original QF")
    m = param.Number(default=1, bounds=(1, 1_000), step=1, label="Donation Amount")
    b = param.Number(default=2, bounds=(1, 5), step=0.1, label="Boost Factor")
    
    def calculate_delta_F(self, m, b):
        return 2 * np.sqrt(self.F) * np.sqrt(m) * (np.sqrt(b) - 1) + m * (b - 1)
    
    def calculate_ratio(self, m, b):
        F_prime = (np.sqrt(self.F) + np.sqrt(m))**2
        F_double_prime = (np.sqrt(self.F) + np.sqrt(m)*np.sqrt(b))**2
        return 100*(F_double_prime - F_prime) / self.F
    
    @param.depends('F', watch=True)  # Watch only the F parameter for the heatmap
    def heatmap(self):
        m_range = np.linspace(*self.param['m'].bounds, 100)
        b_range = np.linspace(*self.param['b'].bounds, 100)
        m_grid, b_grid = np.meshgrid(m_range, b_range)
        ratio_grid = np.vectorize(self.calculate_ratio)(m_grid, b_grid)
        df = pd.DataFrame({'m': m_grid.ravel(), 'b': b_grid.ravel(), 'ratio': ratio_grid.ravel()})
        heatmap = df.hvplot.heatmap(
            title="Percentage Impact of Boost",
            xlim=self.param['m'].bounds,
            ylim=self.param['b'].bounds,
            width=1000,
            height=800,
            xlabel="Donation Amount",
            ylabel="Boost Factor",
            clabel="Percentage Impact of Boosting",
            x='m', y='b', C='ratio', cmap='viridis', colorbar=True)
        return heatmap

    @param.depends('m', 'b', watch=True)  # Watch m and b parameters for the point
    def plot_point(self):
        return hv.Points([(self.m, self.b)]).opts(size=8, color='red')

    @param.depends('F', 'm', 'b')
    def view(self):
        return self.heatmap() * self.plot_point()

    @pn.depends('F', 'm', 'b')
    def report(self):
        delta_F = self.calculate_delta_F(self.m, self.b)
        
        # Calculate the square roots
        sqrt_F = np.sqrt(self.F)
        sqrt_m = np.sqrt(self.m)
        sqrt_b = np.sqrt(self.b)
        
        explanation = f"""

        Original Total Quadradic Funding Without Contribution $$m$$:  
        $$ F = \${self.F:.2f}$$  

        Total Quadradic Funding With Contribution $$m$$:  
        $$ m = \${self.m:.2f} $$  
        $$ F' = (\\sqrt{{F}} + \\sqrt{{m}})^2 $$  
        $$ F' = ${(sqrt_F + sqrt_m)**2:.2f}$$  
        $$ F'/F = {(sqrt_F + sqrt_m)**2/self.F:.2f}$$  

        Total Quadradic Funding With Boosted Contribution $$m*b$$:  
        $$ b = {self.b:.2f} $$  
        $$ F'' = $(\\sqrt{{F}} + \\sqrt{{m}}\*\\sqrt{{b}})^2 $$  
        $$ F'' = ${(sqrt_F + sqrt_m*sqrt_b)**2:.2f}$$  
        $$ F''/F = {(sqrt_F + sqrt_m*sqrt_b)**2/self.F:.2f}$$  

        Impact of the Boosting:  
        $$ (F''-F')/F = {100*((sqrt_F + sqrt_m*sqrt_b)**2-(sqrt_F + sqrt_m)**2)/self.F:.0f}$$%  
        """
        return pn.pane.Markdown(explanation, width=400, styles={'text-align': "center"})

qf = QuadraticFunding()
pn.Row(qf.param, qf.view, qf.report).servable()

