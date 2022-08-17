import medeq
import numpy as np
import pandas as pd
from pyarrow import feather


'''How to print outputs to terminal while also logging them in a saved file:

    python discover.py | tee discover.log

'''


# Import Dataframe
data = pd.read_feather('gb_particledata')
data = data[data['size'] == 2]
results = data.drop(['size'], axis=1)
results = results.dropna().reset_index(drop=True)

# Create DataFrame of MED free parameters and their bounds
parameters = medeq.create_parameters(
    results.columns[:-1],  # Extract the outputs from the dataframe
    minimums=results.iloc[:, :-1].min(),
    maximums=results.iloc[:, :-1].max(),
)

# Create MED object, keeping track of free parameters, samples and results
med = medeq.MED(parameters,
                response_names=["NumberofParticles"],
                seed=123)

# Add previous / manually-evaluated responses
med.augment(
    results.iloc[:, :-1],       # Parameter combinations
    results.iloc[:, -1],        # Output found (i.e. last column)
)

# Save all results to disk - you can load them on another machine
med.save("med_results")

# Discover underlying equation; tell MED what operators it may use
med.discover(
    binary_operators = [
        "+", "-", "*", "/",
        "powa(x, y)=abs(x)^y",
        # "log_abs(x, y)=log(abs(x), abs(y))",
    ],
    # unary_operators = ["cos"],
    # max_size=30,
    # denoise = True,
)
