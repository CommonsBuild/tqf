import pandas as pd
import panel as pn

# Ensure Panel extension is loaded
pn.extension('tabulator')

# Example DataFrame
data = {'Column1': [1, 2, 3], 'Column2': [4, 5, 6], 'Column3': [7, 8, 9]}
df = pd.DataFrame(data)

# Define widths for numeric columns (as an example)
numeric_widths = {'Column1': 100, 'Column2': 150, 'Column3': 200}

# Define Custom CSS to Rotate Column Headers for the specific class
css = """
.tabulator .tabulator-header .tabulator-col .tabulator-col-content .tabulator-col-title {
    transform: rotate(-45deg);
    transform-origin: center;
    text-align: left;
    vertical-align: middle;
}
"""
# Define Tabulator widget with a unique theme class
tabulator = pn.widgets.Tabulator(
    df,
    widths=numeric_widths,
    stylesheets=[css],
)

# Display the Tabulator
tabulator.servable()
