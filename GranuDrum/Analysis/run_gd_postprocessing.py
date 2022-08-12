import numpy as np
import pickle
import pandas as pd
import pyarrow.feather as feather
import os
import gd_postprocessing as pp


def load(path):
    with open(path, "rb") as f:
        obj = pickle.load(f)
    return obj


root = "/rds/projects/w/windowcr-granutools-engd/DigitalTwinStudy/Simulations"
folder = 'GranuDrum'
parameterspath = f"{root}/Generated/parameters_unique.pickle"
df_savepath = 'savedata'

parameters = load(parameterspath)

# Print unique parameter values
print("-" * 80 + "\nUnique Parameter Values:")
for k, v in parameters.items():
    print(f"{k}: {v}")
print("-" * 80)

# Setup empty arrays
size = []
res = []
slid = []
roll = []
coh = []

dynamic_angle = []
cohesive_index = []
poly_3 = []
poly_5 = []

# Extract data
psds = ["large"]
rpm = ['rpm30']
for i, psd in enumerate(psds):
    i = i + 2
    for j, e in enumerate(parameters["restitution"]):
        for k, s in enumerate(parameters["sliding"]):
            for l, r in enumerate(parameters["rolling"]):
                for m, c in enumerate(parameters["cohesion"]):
                    for q, y in enumerate(rpm):
                        try:
                            print(f'Reading data from: gd_{i}_{j}_{k}_{l}_{m}')
                            simdir = f"{root}/{folder}/gd_{i}_{j}_{k}_{l}_{m}/{y}/results"

                            cohesiveindex_array = []
                            dynamicangle_array = []
                            for o in range(5):
                                interface_data = pp.process_images(simdir, 'gd_rtd_', n=50)
                                data = pp.dynamic_angle_of_repose(interface_data)
                                ci = pp.cohesive_index(interface_data, data.averaged_interface)

                                cohesiveindex_array.append(ci)
                                dynamicangle_array.append(data.dynamic_angle_degrees)

                            polynomial = pp.fit_polynomial(data.averaged_interface)
                            polynomial_5th = pp.fit_polynomial(data.averaged_interface, order=5)
                            avgci = np.mean(cohesiveindex_array)
                            avg_da = np.mean(dynamicangle_array)
                            print(f'Success!')
                        except:
                            print('Error reading data')
                            avgci = np.NAN
                            avg_da = np.NAN
                            polynomial = np.NAN
                            polynomial_5th = np.NAN

                        size.append(i)
                        res.append(j)
                        slid.append(k)
                        roll.append(l)
                        coh.append(m)
                        cohesive_index.append(avgci)
                        dynamic_angle.append(avg_da)
                        poly_3.append(polynomial.convert().coef)
                        poly_5.append(polynomial_5th.convert().coef)

# Convert to dataframe and save
data = {'size': size,
        'restitution': res,
        'slidingfriction': slid,
        'rollingfriction': roll,
        'cohesion': coh,
        'y_data': cohesive_index,
        'dynamicangleofrepose': dynamic_angle,
        '3rdorderpolynomial': poly_3,
        '5thorderpolynomial': poly_5
        }
df = pd.DataFrame(data)

if not os.path.isdir(df_savepath):
    os.makedirs(df_savepath)
df.to_feather(f'{df_savepath}/gd_data')
