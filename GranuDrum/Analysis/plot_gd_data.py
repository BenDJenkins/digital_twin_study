import numpy as np
import pandas as pd
import pyarrow.feather as feather
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly

# User dependent variables
filepath = 'data/gd_data_large_30rpm'
rolling_titles = ['Rolling friction: 0.0', 'Rolling friction: 0.001', 'Rolling friction: 0.01',
                  'Rolling friction: 0.1', 'Rolling friction: 0.2', 'Rolling friction: 0.4', 'Rolling friction: 0.7']
slidingfrictionvals = [0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.8]
rollingfrictionvals = [0, 0.001, 0.01, 0.1, 0.2, 0.4, 0.7]

# Load data
df = pd.read_feather(filepath, columns=None, use_threads=True)

# Filter data for plotting
filtered_data = df[df['size'] == 2]
filtered_data = filtered_data[filtered_data['restitution'] == 1]

# Make plots
cols = plotly.colors.DEFAULT_PLOTLY_COLORS

fig = make_subplots(
    rows=1, cols=7,
    subplot_titles=rolling_titles,
    shared_yaxes=True)

for r in range(7):
    for s in range(7):
        focus = (filtered_data.rollingfriction == r) & (filtered_data.slidingfriction == s)
        cohesion = [0, 5000, 10000, 20000, 40000, 70000, 100000]
        cohesiveindex = filtered_data.cohesiveindex[focus]

        if r == 0:
            fig.add_trace(go.Scatter(name=f'Sliding friction: {slidingfrictionvals[s]}',
                                     x=cohesion,
                                     y=cohesiveindex,
                                     line=dict(width=4, dash='dash', color=cols[s]),
                                     marker=dict(color=cols[s]),
                                     legendgroup='group1'
                                     ),
                          row=1,
                          col=r+1)
        else:
            fig.add_trace(go.Scatter(x=cohesion,
                                     y=cohesiveindex,
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

fig.show()
