#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : granubeaker.py
# License: GNU v3.0
# Author : Ben Jenkins <bdj746@student.bham.ac.uk>
# Date   : 17.02.2022


import os

import numpy as np
import coexist


# Relevant paths
script_path = "liggghts_script.sim"

print((
    "\nRunning GranuBeaker simulation:\n"
    f"  Current Directory: {os.getcwd()}\n"
    f"  Simulation Script: {script_path}\n"
), flush = True)


# Load modified simulation script
sim = coexist.LiggghtsSimulation(script_path, verbose=True)


# Run simulation up to given time (s)
line = "\n" + "-" * 80 + "\n"

print(line + "Pouring particles and letting them settle" + line)
sim.step_time(2.0)

sim.execute_command("delete_atoms region 1")


# Extract particle properties as NumPy arrays
time = sim.time()
radii = sim.radii()
positions = sim.positions()
velocities = sim.velocities()

print("\n\n" + line)
print(f"Simulation time: {time} s\nParticle positions:\n{positions}")
print(line + "\n\n")


# Save results as efficient binary NPY-formatted files
np.save(f"results/radii.npy", radii)
np.save(f"results/positions.npy", positions)
np.save(f"results/velocities.npy", velocities)
