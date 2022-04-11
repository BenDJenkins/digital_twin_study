#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : gb_launch_failed.py
# License: GNU v3.0
# Author : Andrei Leonard Nicusan <a.l.nicusan@bham.ac.uk>
# Date   : 16.02.2022
# Adapted by : Ben Jenkins <BDJ746@student.bham.ac.uk>
# Adapted date : 11.04.2022


'''Launch `batch_single.sh` with the correct command-line arguments to run a
single GranuBeaker simulation.
'''


import os
import re
import pickle
import time
import glob

import numpy as np


def load(path):
    with open(path, "rb") as f:
        obj = pickle.load(f)
    return obj


root = "/rds/projects/w/windowcr-granutools-engd/DigitalTwinStudy/Simulations"
parameterspath = f"{root}/Generated/parameters_unique.pickle"
save_missing = True
driverpath = f"{root}/Templates/granubeaker.py"


parameters = load(parameterspath)

# Print unique parameter values
print("-" * 80 + "\nUnique Parameter Values:")
for k, v in parameters.items():
    print(f"{k}: {v}")
print("-" * 80)


finder = re.compile(r"slurm_[0-9]+.stats")
failed = []
skipped = []
unfinished = []

# for i, p in enumerate(parameters["psd"]):
for i in [0]:  # for i in [2, 1, 0]:
    for j, e in enumerate(parameters["restitution"]):
        for k, s in enumerate(parameters["sliding"]):
            for l, r in enumerate(parameters["rolling"]):
                for m, c in enumerate(parameters["cohesion"]):

                    simdir = f"{root}/GranuBeaker/gb_{i}_{j}_{k}_{l}_{m}"

                    files = os.listdir(simdir)
                    slurm_logs = [f for f in files if finder.match(f)]

                    # Run Simulations that weren't run.
                    if len(slurm_logs) == 0:
                        cmd = (
                            f"sbatch --output={simdir}/slurm_%j.out batch_single.sh "
                            f"{driverpath} {simdir}"
                        )
                        print(cmd)
                        os.system(cmd)
                        continue

                    slurm_logs = sorted(slurm_logs)
                    with open(f"{simdir}/{slurm_logs[-1]}") as f:
                        stat = f.read()

                    if "JobState" in stat:
                        if "COMPLETING - Reason None" not in stat:
                            print(f'Simulation {simdir} failed. Rerunning simulation')

                            for filename in glob.glob(f"{simdir}/slurm*"):  # Delete slurm files
                                os.remove(filename)

                            cmd = (
                                f"sbatch --output={simdir}/slurm_%j.out batch_single.sh "
                                f"{driverpath} {simdir}"
                            )
                            print(cmd)
                            os.system(cmd)
                            continue

                    else:
                        print('Completed Simulation')
                        continue

                    time.sleep(0.2)

#
# print(f"{len(failed)} simulations failed:")
# if len(failed):
#     print(f"prefix: {os.path.split(failed[0])[0]}")
#     for f in failed:
#         print(os.path.split(f)[1])
#
# print(f"\n\nSimulations not executed yet:\n{[os.path.split(s)[1] for s in skipped]}")
# print(f"\n\nSimulations not finished yet:\n{[os.path.split(u)[1] for u in unfinished]}")
#
#
# if save_missing and len(failed):
#     file = "failed.txt"
#     print(f"Saving {len(failed)} failed directory names to `{file}`")
#
#     with open(file, "w") as f:
#         for fdir in failed:
#             f.write(f"{fdir}\n")
