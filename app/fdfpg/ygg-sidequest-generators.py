#!/usr/bin/env python
# coding: utf-8

# ## Flexible Design for Funding Public Goods

# In[1]:


import numpy as np

# Number of Citizens in the Society
N = 30

# Society is a set of citizens
society = set(range(N))

# Community is a random subset of the society. The community size is from 1 up to 1/2 of the society.
community = np.random.choice(a=list(society), size=np.random.randint(1, len(society)/2), replace=False, p=None)

# Public Goods are proposed by community members. Cardinality is from 1 up to 1/2 size of the community.
public_goods = list(enumerate(np.random.choice(a=list(community), size=np.random.randint(1, len(community)/2), replace=True, p=None)))


# ## 3.1 Side Quest: Generating Value Functions

# ### Polynomial Generator.

# In[2]:


c1 = {'exponent': 1,
 'f0': 0.2,
 'f1': 0.8,
 'initial_slope': 1,
 'name': 'CurveGenerator18499',
 'num_oscillations': 2}


# In[3]:


import param
import numpy as np
import pandas as pd
import hvplot.pandas
import panel as pn

class CurveGenerator(param.Parameterized):
    f0 = param.Number(default=0.2, bounds=(0, 1), doc="Value of f(0)")
    f1 = param.Number(default=0.8, bounds=(0, 1), doc="Value of f(1)")
    initial_slope = param.Number(default=1, bounds=(-5, 5), doc="Initial slope of the curve")
    exponent = param.Number(default=1, bounds=(1, 5), doc="Exponent of the curve")
    num_oscillations = param.Integer(default=1, bounds=(0, 5), doc="Number of oscillations/peaks in the curve")
    
    @param.depends('f0', 'f1', 'initial_slope', 'exponent', 'num_oscillations')
    def curve(self, x):
        epsilon = 1e-10
        b = self.f0
        a = self.initial_slope / (self.exponent * (b + epsilon)**(self.exponent-1))
        c = (self.f1 - self.f0 - a) / 2
        d = self.num_oscillations
        y = a*x**self.exponent + b + c*np.sin(d*np.pi*x)
        
        # Scale and shift the curve to ensure it starts at f0 and ends at f1
        y = self.f0 + (self.f1 - self.f0) * (y - y.min()) / (y.max() - y.min())
        
        return y
    
    @param.depends('f0', 'f1', 'initial_slope', 'exponent', 'num_oscillations')
    def view(self):
        x = np.linspace(0, 1, 100)
        y = self.curve(x)
        df = pd.DataFrame({'x': x, 'y': y})
        return df.hvplot.line(x='x', y='y', ylim=(0, 1.01))

curve_gen = CurveGenerator(**c1)
pn.Row(curve_gen.param, curve_gen.view).servable()


# In[4]:


curve_gen.param.values()


# ### Sigmoid Generator

# In[5]:


s1 = {'exponent': 0.3,
 'f0': 0,
 'f1': 0.5,
 'initial_slope': -5,
 'name': 'SigmoidGenerator03300',
 'oscillations': 2}

s2 = {'exponent': 0.4,
 'f0': 0,
 'f1': 0.5,
 'initial_slope': 4.4,
 'name': 'SigmoidGenerator03300',
 'oscillations': 2}


# In[6]:


import numpy as np
import pandas as pd
import panel as pn
import param
import hvplot.pandas

class SigmoidGenerator(param.Parameterized):
    f0 = param.Number(default=0.5, bounds=(0, 1), doc="Value of the function at x=0")
    f1 = param.Number(default=0.5, bounds=(0, 1), doc="Value of the function at x=1")
    initial_slope = param.Number(default=1, bounds=(-5, 5), doc="Initial slope of the curve")
    exponent = param.Number(default=0.3, bounds=(0.1, 0.5), doc="Exponent of the curve")
    oscillations = param.Integer(default=1, bounds=(1, 5), doc="Number of oscillations/peaks in the curve")
    
    @param.depends('f0', 'f1', 'initial_slope', 'exponent', 'oscillations')
    def view(self):
        x = np.linspace(0, 1, 400)
        y = self.f0 + (self.f1 - self.f0) / (1 + np.exp(-self.initial_slope * (x - 0.5) * 10))**self.exponent
        y = y + 0.1 * np.sin(self.oscillations * np.pi * x)
        
        # Clip y values to ensure they stay within [0, 1]
        y = np.clip(y, 0, 1)
        
        df = pd.DataFrame({'x': x, 'y': y})
        return df.hvplot.line(x='x', y='y', ylim=(-0.01, 1.01))

sigmoid_gen = SigmoidGenerator(**s2)
pn.Row(sigmoid_gen.param, sigmoid_gen.view).servable()


# In[7]:


sigmoid_gen.param.values()


# In[8]:


p1 = {'exponent_param': 0.2,
 'f0': 0.1,
 'f1': 0.8,
 'name': 'PowerFunctionGenerator22564'}


# ### Power Function Generator

# In[9]:


import numpy as np
import pandas as pd
import panel as pn
import param
import hvplot.pandas

class PowerFunctionGenerator(param.Parameterized):
    f0 = param.Number(default=0.1, bounds=(0, 1), doc="Value of the function at x=0")
    f1 = param.Number(default=0.5, bounds=(0, 1), doc="Value of the function at x=1")
    exponent_param = param.Number(default=0.5, bounds=(0.1, 2), doc="Parameter determining the exponent and slope")
    
    @param.depends('f0', 'f1', 'exponent_param')
    def view(self):
        epsilon = 1e-10

        x = np.linspace(0.001, 1, 400)  # Start from 0.001 to avoid division by zero
        
        # Calculate the exponent based on the provided parameter
        b = 2 * self.exponent_param  # This maps [0, 1] to [0, 2] for the exponent
        
        # Using the conditions f(0) = f0 and f(1) = f1 to solve for 'a' and 'c'
        a = self.f0
        c = (self.f1 - self.f0) / (1 ** b - 0 ** b + epsilon)
        
        y = a + c * x ** b
        
        # Clip y values to ensure they stay within [0, 1]
        y = np.clip(y, 0, 1)
        
        df = pd.DataFrame({'x': x, 'y': y})
        return df.hvplot.line(x='x', y='y', ylim=(0, 1.01))

power_func_gen = PowerFunctionGenerator(**p1)
pn.Row(power_func_gen.param, power_func_gen.view).servable()


# In[10]:


power_func_gen.param.values()


# ### Generating the Generators with NumberGen and Param.

# In[11]:


import numbergen as ng

# For CurveGenerator
polynomial_curve_generator_params = dict(
    f0=ng.UniformRandom(lbound=0, ubound=1)(),
    f1=ng.UniformRandom(lbound=0, ubound=1)(),
    initial_slope=ng.UniformRandom(lbound=-5, ubound=5)(),
    exponent=ng.UniformRandom(lbound=1, ubound=5)(),
    num_oscillations=int(ng.UniformRandom(lbound=0, ubound=5)())
)

# For SigmoidGenerator
sigmoid_curve_generator_params = dict(
    f0=ng.UniformRandom(lbound=0, ubound=1)(),
    f1=ng.UniformRandom(lbound=0, ubound=1)(),
    initial_slope=ng.UniformRandom(lbound=-5, ubound=5)(),
    exponent=ng.UniformRandom(lbound=0.1, ubound=0.5)(),
    oscillations=int(ng.UniformRandom(lbound=1, ubound=5)())
)

# For PowerFunctionGenerator
power_curve_generator_params = dict(
    f0=ng.UniformRandom(lbound=0, ubound=0.5)(),
    f1=ng.UniformRandom(lbound=0, ubound=1)(),
    exponent_param=ng.UniformRandom(lbound=0.1, ubound=2)()
)

# Now, you can use these dictionaries to create instances of your classes as you've done in your code.


# ### Polynomial Value Function Generator

# In[12]:


# Instantiate CurveGenerator with polynomial_curve_generator_params
polynomial_curve_gen_instance = CurveGenerator(**polynomial_curve_generator_params)

polynomial_curve_gen_instance.view()


# ### Sigmoid Vlue Function Generator

# In[13]:


# Instantiate SigmoidGenerator with sigmoid_curve_generator_params
sigmoid_curve_gen_instance = SigmoidGenerator(**sigmoid_curve_generator_params)
sigmoid_curve_gen_instance.view()


# ### Power Function Value Function Generator

# In[14]:


# Instantiate PowerFunctionGenerator with power_curve_generator_params
power_curve_gen_instance = PowerFunctionGenerator(**power_curve_generator_params)
power_curve_gen_instance.view()


# In[15]:


value_generators = [
    (CurveGenerator, polynomial_curve_generator_params),
    (SigmoidGenerator, sigmoid_curve_generator_params),
    (PowerFunctionGenerator, power_curve_generator_params),
]

# Generate a random index
index = np.random.choice(len(value_generators))

# Use the index to select an item from value_generators
selected_generator = value_generators[index]


# In[16]:


index


# In[17]:


selected_generator


# In[18]:


selected_generator[0]


# In[19]:


selected_generator[0](**selected_generator[1])


# In[20]:


import numbergen as ng
import numpy as np


# For CurveGenerator
polynomial_curve_generator_params = dict(
    f0=ng.UniformRandom(lbound=0, ubound=1)(),
    f1=ng.UniformRandom(lbound=0, ubound=1)(),
    initial_slope=ng.UniformRandom(lbound=-5, ubound=5)(),
    exponent=ng.UniformRandom(lbound=1, ubound=5)(),
    num_oscillations=int(ng.UniformRandom(lbound=0, ubound=5)())
)

# For SigmoidGenerator
sigmoid_curve_generator_params = dict(
    f0=ng.UniformRandom(lbound=0, ubound=1)(),
    f1=ng.UniformRandom(lbound=0, ubound=1)(),
    initial_slope=ng.UniformRandom(lbound=-5, ubound=5)(),
    exponent=ng.UniformRandom(lbound=0.1, ubound=0.5)(),
    oscillations=int(ng.UniformRandom(lbound=1, ubound=5)())
)

# For PowerFunctionGenerator
power_curve_generator_params = dict(
    f0=ng.UniformRandom(lbound=0, ubound=1)(),
    f1=ng.UniformRandom(lbound=0, ubound=1)(),
    exponent_param=ng.UniformRandom(lbound=0.1, ubound=2)()
)

# Now, you can use these dictionaries to create instances of your classes as you've done in your code.
value_generators = np.array([
    (CurveGenerator, polynomial_curve_generator_params),
    (SigmoidGenerator, sigmoid_curve_generator_params),
    (PowerFunctionGenerator, power_curve_generator_params),
])

# Generate a random array of indices of length n
samples = np.random.choice(len(value_generators), size=len(public_goods)*len(society))

# Use numpy's advanced indexing to obtain the selected_generators
sampled_generators = value_generators[samples]

# Instantiate utility curves using python param and numbergen
sampled_utility = [Generator(**params) for Generator, params in sampled_generators]

sampled_utility[:5]



# In[21]:


import pandas as pd

pd.DataFrame([s.param.values() for s in sampled_utility])

sample_p_i = pn.widgets.IntSlider(name='Utility Value Function', start=0, end=len(sampled_utility)-1)

pn.Row(sample_p_i, pn.bind(lambda i: sampled_utility[i].view(), i=sample_p_i))#.param.value_throttled))


# ## 3.1 Side Quest: Generating Value Functions - Continued

# In[22]:


import param
import numpy as np
import pandas as pd
import hvplot.pandas
import panel as pn

class CurveGenerator(param.Parameterized):
    f0 = param.Number(default=0.2, bounds=(0, 1), doc="Value of f(0)")
    f1 = param.Number(default=0.8, bounds=(0, 1), doc="Value of f(1)")
    initial_slope = param.Number(default=1, bounds=(-5, 5), doc="Initial slope of the curve")
    exponent = param.Number(default=1, bounds=(1, 5), doc="Exponent of the curve")
    num_oscillations = param.Integer(default=1, bounds=(0, 5), doc="Number of oscillations/peaks in the curve")
    
    def x(self):
        return np.linspace(0, 1, 400)
    
    def f(self, x):
        epsilon = 1e-10
        b = self.f0
        a = self.initial_slope / (self.exponent * (b + epsilon)**(self.exponent-1))
        c = (self.f1 - self.f0 - a) / 2
        d = self.num_oscillations
        y = a*x**self.exponent + b + c*np.sin(d*np.pi*x)
        
        # Scale and shift the curve to ensure it starts at f0 and ends at f1
        y = self.f0 + (self.f1 - self.f0) * (y - y.min()) / (y.max() - y.min())
        return y
    
    @param.depends('f0', 'f1', 'initial_slope', 'exponent', 'num_oscillations')
    def view(self):
        x = self.x()
        y = self.f(x)
        df = pd.DataFrame({'x': x, 'y': y})
        return df.hvplot.line(x='x', y='y', ylim=(0, 1.01), width=500, height=400)

class SigmoidGenerator(param.Parameterized):
    f0 = param.Number(default=0.5, bounds=(0, 1), doc="Value of the function at x=0")
    f1 = param.Number(default=0.5, bounds=(0, 1), doc="Value of the function at x=1")
    initial_slope = param.Number(default=1, bounds=(-5, 5), doc="Initial slope of the curve")
    exponent = param.Number(default=0.3, bounds=(0.1, 0.5), doc="Exponent of the curve")
    oscillations = param.Integer(default=1, bounds=(1, 5), doc="Number of oscillations/peaks in the curve")
    
    def x(self):
        return np.linspace(0, 1, 400)
    
    def f(self, x):
        y = self.f0 + (self.f1 - self.f0) / (1 + np.exp(-self.initial_slope * (x - 0.5) * 10))**self.exponent
        y = y + 0.1 * np.sin(self.oscillations * np.pi * x)
        
        # Clip y values to ensure they stay within [0, 1]
        y = np.clip(y, 0, 1)
        return y
    
    @param.depends('f0', 'f1', 'initial_slope', 'exponent', 'oscillations')
    def view(self):
        x = self.x()
        y = self.f(x)
        df = pd.DataFrame({'x': x, 'y': y})
        return df.hvplot.line(x='x', y='y', ylim=(-0.01, 1.01))

class PowerFunctionGenerator(param.Parameterized):
    f0 = param.Number(default=0.1, bounds=(0, 1), doc="Value of the function at x=0")
    f1 = param.Number(default=0.5, bounds=(0, 1), doc="Value of the function at x=1")
    exponent_param = param.Number(default=0.5, bounds=(0.1, 2), doc="Parameter determining the exponent and slope")
    
    def x(self):
        return np.linspace(0.001, 1, 400)  # Start from 0.001 to avoid division by zero
    
    def f(self, x):
        epsilon = 1e-10
        b = 2 * self.exponent_param  # This maps [0, 1] to [0, 2] for the exponent
        a = self.f0
        c = (self.f1 - self.f0) / (1 ** b - 0 ** b + epsilon)
        y = a + c * x ** b
        
        # Clip y values to ensure they stay within [0, 1]
        y = np.clip(y, 0, 1)
        return y
    
    @param.depends('f0', 'f1', 'exponent_param')
    def view(self):
        x = self.x()
        y = self.f(x)
        df = pd.DataFrame({'x': x, 'y': y})
        return df.hvplot.line(x='x', y='y', ylim=(0, 1.01))


# In[23]:


import numbergen as ng
import numpy as np


# For CurveGenerator
def polynomial_curve_generator_params():
    return dict(
        f0=ng.UniformRandom(lbound=0, ubound=0.5, seed=None)(),
        f1=ng.UniformRandom(lbound=0, ubound=1)(),
        initial_slope=ng.UniformRandom(lbound=-5, ubound=5)(),
        exponent=ng.UniformRandom(lbound=1, ubound=5)(),
        num_oscillations=int(ng.UniformRandom(lbound=0, ubound=5)())
    )

# For SigmoidGenerator
def sigmoid_curve_generator_params():
    return dict(
        f0=ng.UniformRandom(lbound=0, ubound=0.5)(),
        f1=ng.UniformRandom(lbound=0, ubound=1)(),
        initial_slope=ng.UniformRandom(lbound=-5, ubound=5)(),
        exponent=ng.UniformRandom(lbound=0.1, ubound=0.5)(),
        oscillations=int(ng.UniformRandom(lbound=1, ubound=5)())
    )

# For PowerFunctionGenerator
def power_curve_generator_params():
    return dict(
        f0=ng.UniformRandom(lbound=0, ubound=0.5)(),
        f1=ng.UniformRandom(lbound=0, ubound=1)(),
        exponent_param=ng.UniformRandom(lbound=0.1, ubound=2)()
)



# Now, you can use these dictionaries to create instances of your classes as you've done in your code.
value_function_generators = np.array([
    (CurveGenerator, polynomial_curve_generator_params),
    (SigmoidGenerator, sigmoid_curve_generator_params),
    (PowerFunctionGenerator, power_curve_generator_params),
])

# Use numpy's advanced indexing to obtain the selected_generators
value_function_samples= value_function_generators[np.random.choice(len(value_function_generators), size=len(public_goods)*len(society))]

# Instantiate utility curves using python param and numbergen
value_functions = [Generator(**params()) for Generator, params in value_function_samples]


# In[24]:


df_value_functions = pd.DataFrame([s.f(s.x()) for s in value_functions]).T
df_value_functions.columns = [(p, i) for p in public_goods for i in society]
df_value_functions.columns.name = "value_p_i"
df_value_functions.index = np.linspace(0,1,len(df_value_functions))
df_value_functions.index.name = "funding"


# In[25]:


df_value_functions


# In[26]:


df_value_functions.hvplot.line(x='funding', color='blue', alpha=0.2, line_width=5)


# In[27]:


df_value_functions.melt(ignore_index=False)


# In[28]:


df_value_functions_melted = df_value_functions.melt(ignore_index=False)
df_value_functions_melted


# In[29]:


df_value_functions_melted['public_good'] = df_value_functions_melted['value_p_i'].astype(str).apply(eval).apply(lambda x: x[0]).astype(str)
df_value_functions_melted['citizen'] = df_value_functions_melted['value_p_i'].astype(str).apply(eval).apply(lambda x: x[1]).astype(str)


# In[30]:


df_value_functions_melted


# In[31]:


import hvplot.pandas


# In[32]:


df_value_functions_melted.hvplot.scatter(y='value', by='public_good', alpha=0.1)


# In[33]:


mean_utility_df = df_value_functions_melted.groupby(['funding', 'public_good'])[['value']].mean().reset_index()


# In[34]:


mean_utility_df.hvplot.scatter(y='value', by='public_good')


# In[35]:


mean_utility_df


# In[36]:


df_value_functions


# In[37]:


import param
import numpy as np
import panel as pn
import hvplot.pandas
import pandas as pd

class ConcaveFunctionGenerator(param.Parameterized):
    f0 = param.Number(default=0.2, bounds=(0, 1), doc="Value of f(0)")
    f1 = param.Number(default=0.8, bounds=(0, 1), doc="Value of f(1)")
    slope = param.Number(default=10, bounds=(1, 50), doc="Slope of the curve")

    @param.depends('f0', 'f1', 'slope')
    def f(self, x):
        # Using the sigmoid function as a base
        y = 1 / (1 + np.exp(-self.slope * (x - 0.5)))
        
        # Adjusting the function to start at f0 and end at f1
        y = self.f0 + (self.f1 - self.f0) * (y - y.min()) / (y.max() - y.min())
        
        return y

    @param.depends('f0', 'f1', 'slope')
    def view(self):
        x = np.linspace(0, 1, 400)
        y = self.f(x)
        df = pd.DataFrame({'x': x, 'y': y})
        return df.hvplot.line(x='x', y='y', ylim=(0, 1.01), width=500, height=400)

concave_gen = ConcaveFunctionGenerator()
pn.Row(concave_gen.param, concave_gen.view).servable()


# In[38]:


import param
import numpy as np
import panel as pn
import hvplot.pandas
import pandas as pd

class ConcaveFunctionGenerator(param.Parameterized):
    f0 = param.Number(default=0.2, bounds=(0, 1), doc="Value of f(0)")
    f1 = param.Number(default=0.8, bounds=(0, 1), doc="Value of f(1)")
    steepness = param.Number(default=5, bounds=(1, 20), doc="Steepness of the curve")

    @param.depends('f0', 'f1', 'steepness')
    def f(self, x):
        # Using the negative exponential function as a base
        y = 1 - np.exp(-self.steepness * x)
        
        # Adjusting the function to start at f0 and end at f1
        y = self.f0 + (self.f1 - self.f0) * (y - y.min()) / (y.max() - y.min())
        
        return y

    @param.depends('f0', 'f1', 'steepness')
    def view(self):
        x = np.linspace(0, 1, 400)
        y = self.f(x)
        df = pd.DataFrame({'x': x, 'y': y})
        return df.hvplot.line(x='x', y='y', ylim=(0, 1.01), width=500, height=400)

concave_gen = ConcaveFunctionGenerator()
pn.Row(concave_gen.param, concave_gen.view).servable()


# In[39]:


import param
import numpy as np
import panel as pn
import hvplot.pandas
import pandas as pd

class ConcaveFunctionGenerator(param.Parameterized):
    f0 = param.Number(default=0.2, bounds=(0, 1), doc="Value of f(0)")
    f1 = param.Number(default=0.8, bounds=(0, 1), softbounds=(0, 1), doc="Value of f(1)")
    steepness = param.Number(default=5, bounds=(1, 20), doc="Steepness of the curve")

    def __init__(self, **params):
        super().__init__(**params)
        self._update_f1_bounds()

    @param.depends('f0', watch=True)
    def _update_f1_bounds(self):
        # Clip the value of f1 if it's below f0
        self.f1 = max(self.f0, self.f1)
        
        # Update the lower bound of f1 to be the value of f0
        self.param['f1'].bounds = (self.f0, 1)
        
    def x(self):
        return np.linspace(0, 1, 400)

    @param.depends('f0', 'f1', 'steepness')
    def f(self, x):
        # Using the negative exponential function as a base
        y = 1 - np.exp(-self.steepness * x)
        
        # Adjusting the function to start at f0 and end at f1
        y = self.f0 + (self.f1 - self.f0) * (y - y.min()) / (y.max() - y.min())
        
        return y

    @param.depends('f0', 'f1', 'steepness')
    def view(self):
        x = self.x()
        y = self.f(x)
        df = pd.DataFrame({'x': x, 'y': y})
        return df.hvplot.line(x='x', y='y', ylim=(0, 1.01))

concave_gen = ConcaveFunctionGenerator()
pn.Row(concave_gen.param, concave_gen.view).servable()


# In[40]:


ConcaveFunctionGenerator(f0=1,f1=0)


# In[41]:


import numbergen as ng
import numpy as np


# For CurveGenerator
def concave_function_generator():
    return dict(
        f0=ng.BoundedNumber(generator=ng.NormalRandom(mu=0.1, sigma=0.2), bounds=(0,1))(),
        f1=ng.BoundedNumber(generator=ng.NormalRandom(mu=0.5, sigma=0.3), bounds=(0,1))(),
        steepness=ng.UniformRandom(lbound=1, ubound=20)(),
    )


# In[42]:


concave_function_generator()


# In[43]:


value_functions = [ConcaveFunctionGenerator(**concave_function_generator()) for p_i in range(len(public_goods)*len(society))]


# In[44]:


df_value_functions = pd.DataFrame([s.f(s.x()) for s in value_functions]).T
df_value_functions.columns = [(p, i) for p in public_goods for i in society]
df_value_functions.columns.name = "value_p_i"
df_value_functions.index = np.linspace(0,1,len(df_value_functions))
df_value_functions.index.name = "funding"


# In[45]:


df_value_functions


# In[46]:


import pandas as pd

pd.DataFrame([s.param.values() for s in value_functions])

sample_p_i = pn.widgets.IntSlider(name='Utility Value Function', start=0, end=len(value_functions)-1)

pn.Row(sample_p_i, pn.bind(lambda i: value_functions[i].view(), i=sample_p_i))#.param.value_throttled))


# In[47]:


df_value_functions.hvplot.line(x='funding', color='blue', alpha=0.1, line_width=3)


# In[48]:


df_value_functions['mean'] = df_value_functions.mean(axis=1)
df_value_functions['std'] = df_value_functions.std(axis=1)
df_value_functions['low'] = df_value_functions['mean'] - df_value_functions['std']
df_value_functions['high'] = df_value_functions['mean'] + df_value_functions['std']


# In[49]:


df_value_functions.hvplot.line(y='mean', ylabel='Value to Society') * df_value_functions.hvplot.area(y='low',y2='high', alpha=0.5)


# ## Public Goods Distributions Explorer
# 
# This widget allows us to sample funding distributions F_P. We can explore F_P as a normal, constant, uniform, or exponential distribution.

# In[201]:


public_goods_funding_model = {'constant_value': 0.5,
 'distribution_type': 'exponential',
 'lambda_param': 2.8000000000000003,
 'mean': 0.2,
 'n': len(public_goods),
 'name': 'PublicGoodsFundingDistributionGenerator53483',
 'std_dev': 0.2}


# In[202]:


import param
import numpy as np
import pandas as pd
import panel as pn
import hvplot.pandas

class PublicGoodsFundingDistributionGenerator(param.Parameterized):
    distribution_type = param.ObjectSelector(default="normal", objects=["normal", "constant", "uniform", "exponential"])
    mean = param.Number(default=0.5, bounds=(0, 1))
    n = param.Integer(default=100, bounds=(1, 1000))
    
    # Additional parameters for specific distributions
    std_dev = param.Number(default=0.1, bounds=(0, 0.5))  # for normal distribution
    constant_value = param.Number(default=0.5, bounds=(0, 1))  # for constant distribution
    lambda_param = param.Number(default=1.0, bounds=(0.1, 5))  # for exponential distribution
    
    @param.depends('distribution_type', 'mean', 'n', 'std_dev', 'constant_value', 'lambda_param')
    def generate_distribution(self):
        if self.distribution_type == "normal":
            distribution = np.clip(np.random.normal(self.mean, self.std_dev, self.n), 0, 1)
        elif self.distribution_type == "constant":
            distribution = np.full(self.n, self.constant_value)
        elif self.distribution_type == "uniform":
            distribution = np.random.uniform(0, 1, self.n)
        elif self.distribution_type == "exponential":
            distribution = np.clip(np.random.exponential(1/self.lambda_param, self.n), 0, 1)
        distribution = pd.Series(distribution, name='Public Goods Funding Distribution')
        return distribution / distribution.sum()
        
    
    @param.depends('distribution_type', 'mean', 'n', 'std_dev', 'constant_value', 'lambda_param')
    def view(self):
        data = self.generate_distribution()
        df = pd.DataFrame({'Value': data})
        return df.hvplot.hist('Value', bins=30, title='Public Goods Funding Histogram')

# Create an instance
dist_gen = PublicGoodsFundingDistributionGenerator(**public_goods_funding_model)

# Use panel to render the interactive system
pn.Row(dist_gen.param, dist_gen.view).servable()


# #### Saving State with Params

# In[160]:


dist_gen.param.values()


# In[161]:


dist_gen.generate_distribution()

