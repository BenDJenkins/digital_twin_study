import numpy as np
import pandas as pd
import pyarrow.feather as feather
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly
import seaborn as sns

# Values to plot
size_index = 2
restitution_index = 0

# Relevant paths
filepath = 'data/gd_data_large_30rpm'

# DEM parameter values
rolling_titles = ['Rolling friction: 0.0', 'Rolling friction: 0.001', 'Rolling friction: 0.01',
                  'Rolling friction: 0.1', 'Rolling friction: 0.2', 'Rolling friction: 0.4', 'Rolling friction: 0.7']
slidingfrictionvals = [0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.8]
rollingfrictionvals = [0, 0.001, 0.01, 0.1, 0.2, 0.4, 0.7]
cohesion = [0, 5000, 10000, 20000, 30000, 40000, 50000]
restitutionvals = [0.1, 0.5, 0.9]

# Load data
df = pd.read_feather(filepath, columns=None, use_threads=True)

# Filter data for plotting
filtered_data = df[df['size'] == size_index]
filtered_data = filtered_data[filtered_data['restitution'] == restitution_index]

# Make plots
cols = plotly.colors.DEFAULT_PLOTLY_COLORS

fig = make_subplots(
    rows=1, cols=7,
    subplot_titles=rolling_titles,
    shared_yaxes=True)

for r in range(7):
    for s in range(7):
        focus = (filtered_data.rollingfriction == r) & (filtered_data.slidingfriction == s)
        y_data = filtered_data.dynamicangleofrepose[focus]

        if r == 0:
            fig.add_trace(go.Scatter(name=f'Sliding friction: {slidingfrictionvals[s]}',
                                     x=cohesion,
                                     y=y_data,
                                     line=dict(width=4, dash='dash', color=cols[s]),
                                     marker=dict(color=cols[s]),
                                     legendgroup='group1'
                                     ),
                          row=1,
                          col=r+1)
        else:
            fig.add_trace(go.Scatter(x=cohesion,
                                     y=y_data,
                                     line=dict(width=4, dash='dash', color=cols[s]),
                                     marker=dict(color=cols[s]),
                                     legendgroup='group1',
                                     showlegend=False
                                     ),
                          row=1,
                          col=r+1)

# med_coh = np.arange(0, 100000, 10000)
#
# for r in range(7):
#     for s in range(7):
#         restitution = 0.9
#         slidingfriction = slidingfrictionvals[s]
#         rollingfriction = rollingfrictionvals[r]
#         par = ((14364.258 + (-102.00902 * slidingfriction)) + ((-17.655941 * slidingfriction) * rollingfriction))
#         par_num = [par]*len(med_coh)
#         fig.add_trace(go.Scatter(x=med_coh, y=par_num,
#                       line=dict(color='black')),
#                       row=1,
#                       col=r+1)

fig.update_layout(
    title=f"Variation of Dynamic Angle of Repose with DEM Parameters for largest particle size "
          f"and restitution of {restitutionvals[restitution_index]}.",
    xaxis_title="Cohesive Energy Density",
    yaxis_title="Cohesive Index",
    font=dict(
        size=18,
    )
)
fig.show()
