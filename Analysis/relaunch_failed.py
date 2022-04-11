#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : check_granubeaker.py
# License: GNU v3.0
# Author : Andrei Leonard Nicusan <a.l.nicusan@bham.ac.uk>
# Date   : 16.02.2022


'''Launch `batch_single.sh` with the correct command-line arguments to run a
single GranuBeaker simulation.
'''


import os
import re
import time
import pickle

import numpy as np


def load(path):
    with open(path, "rb") as f:
        obj = pickle.load(f)
    return obj


root = "/rds/projects/w/windowcr-granutools-engd/DigitalTwinStudy/Simulations"
parameterspath = f"{root}/Generated/parameters_unique.pickle"
driverpath = f"{root}/Templates/granubeaker.py"

failedpath = "failed.pickle"


parameters = load(parameterspath)

# Print unique parameter values
print("-" * 80 + "\nUnique Parameter Values:")
for k, v in parameters.items():
    print(f"{k}: {v}")
print("-" * 80)



failed = load(failedpath)

for fdir in failed:

    cmd = (
        f"sbatch --output={fdir}/slurm_%j.out {root}/batch_single.sh "
        f"{driverpath} {fdir}"
    )
    print(cmd)
    os.system(cmd)

    # Wait for half a second before submitting a new job to
    # avoid overflowing SLURM
    time.sleep(0.5)
