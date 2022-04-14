#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : gb_velocity_check.py
# License: GNU v3.0
# Adapted by : Ben Jenkins <BDJ746@student.bham.ac.uk>
# Adapted date : 13.04.2022


import pickle
import numpy as np


def load(path):
    with open(path, "rb") as f:
        obj = pickle.load(f)
    return obj

def check(list1):
    # traverse in the list
    for x in list1:

        # compare with all the values
        # with val
        if 0 > x:

            return False
    return True


root = "/rds/projects/w/windowcr-granutools-engd/DigitalTwinStudy/Simulations"
parameterspath = f"{root}/Generated/parameters_unique.pickle"
driverpath = f"{root}/Templates/granubeaker.py"


parameters = load(parameterspath)

# Print unique parameter values
print("-" * 80 + "\nUnique Parameter Values:")
for k, v in parameters.items():
    print(f"{k}: {v}")
print("-" * 80)


# for i, p in enumerate(parameters["psd"]):
for i in [2]:  # for i in [2, 1, 0]:
    for j, e in enumerate(parameters["restitution"]):
        for k, s in enumerate(parameters["sliding"]):
            for l, r in enumerate(parameters["rolling"]):
                for m, c in enumerate(parameters["cohesion"]):

                    try:
                        simdir = f"{root}/GranuBeaker/gb_{i}_{j}_{k}_{l}_{m}/results"

                        velocities = np.load(f'{simdir}/velocities.npy')

                        sum_velocities = np.sum(velocities)

                        if sum_velocities > 0:
                            print(f'gb_{i}_{j}_{k}_{l}_{m} has velocity sum of {sum_velocities}')

                        else:
                            continue

                    except:
                        print('No data')


