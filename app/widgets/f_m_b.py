import hvplot.pandas
import panel as pn
import param
import numpy as np
import pandas as pd
import holoviews as hv
pn.extension('mathjax')

class QuadraticFunding(param.Parameterized):
    F = param.Number(default=1_000, bounds=(0, 10_000), step=100)
    m = param.Number(default=10, bounds=(1, 1_000), step=1)
    b = param.Number(default=2, bounds=(1, 5), step=0.1)
    
    def calculate_delta_F(self, m, b):
        return 2 * np.sqrt(self.F) * np.sqrt(m) * (np.sqrt(b) - 1) + m * (b - 1)
    
    @param.depends('F', 'm', 'b')
    def view(self):
        m_range = np.linspace(*self.param['m'].bounds, 100)
        b_range = np.linspace(*self.param['b'].bounds, 100)
        m_grid, b_grid = np.meshgrid(m_range, b_range)
        delta_F_grid = np.vectorize(self.calculate_delta_F)(m_grid, b_grid)
        df = pd.DataFrame({'m': m_grid.ravel(), 'b': b_grid.ravel(), 'delta_F': delta_F_grid.ravel()})
        heatmap = df.hvplot.heatmap(x='m', y='b', C='delta_F', cmap='viridis', colorbar=True)
        
        # Add point for current m and b
        point = hv.Points([(self.m, self.b)]).opts(size=10, color='red')
        return heatmap * point


    @pn.depends('F', 'm', 'b')
    def report(self):
        delta_F = self.calculate_delta_F(self.m, self.b)
        
        # Calculate the square roots
        sqrt_F = np.sqrt(self.F)
        sqrt_m = np.sqrt(self.m)
        sqrt_b = np.sqrt(self.b)

        delta_F_prime = (sqrt_F + sqrt_m)**2 - self.F
        delta_F_double_prime = (sqrt_F + sqrt_m*sqrt_b)**2 - self.F
        percentage_boost = ((delta_F_double_prime - delta_F_prime) / delta_F_prime) * 100

        
        # Compute the percentage impacts
        percentage_impact_m_over_F = (delta_F_prime / self.F) * 100
        percentage_impact_mb_over_F = (delta_F_double_prime / self.F) * 100
        percentage_impact_mb_over_m = ((delta_F_double_prime - delta_F_prime) / delta_F_prime) * 100

        
        explanation = f"""

        Original Total Quadradic Funding Without Contribution $$m$$:  
        $$ F = \${self.F:.2f}$$  

        Total Quadradic Funding With Contribution $$m$$:  
        $$ m = \${self.m:.2f} $$  
        $$ F' = (\\sqrt{{F}} + \\sqrt{{m}})^2 $$  
        $$ F' = ${(sqrt_F + sqrt_m)**2:.2f}$$  
        $$ F'\/F = {(sqrt_F + sqrt_m)**2/self.F:.2f}$$  

        Total Quadradic Funding With Boosted Contribution $$m*b$$:  
        $$ b = {self.b:.2f} $$  
        $$ F'' = $(\\sqrt{{F}} + \\sqrt{{m}}\*\\sqrt{{b}})^2 $$  
        $$ F'' = ${(sqrt_F + sqrt_m*sqrt_b)**2:.2f}$$  
        $$ F''\/F = {(sqrt_F + sqrt_m*sqrt_b)**2/self.F:.2f}$$  

        Impact of the Boosting:  
        $$ F''\/F' = {(sqrt_F + sqrt_m*sqrt_b)**2/(sqrt_F + sqrt_m)**2:.2f}$$  
        """
        return pn.pane.Markdown(explanation, width=400, styles={'text-align': "center"})

qf = QuadraticFunding()
pn.Row(qf.param, qf.view, qf.report).servable()

