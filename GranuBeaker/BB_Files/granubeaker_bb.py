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
    "-----------------------------------------------------------------------\n"
    "[Python]\n"
    "Running GranuBeaker simulation:\n"
    f"  Current Directory: {os.getcwd()}\n"
    f"  Simulation Script: {script_path}\n"
    "-----------------------------------------------------------------------\n"
), flush = True)


# Load modified simulation script
sim = coexist.LiggghtsSimulation(script_path, verbose=True)


# Run simulation up to given time (s)
line = "\n" + "-" * 80 + "\n"

# Inserting Particles
print(line + "Pouring particles" + line)
sim.step_time(2.0)

# Allowing particles to settle
print(line + "Letting remaining particles fall and settle" + line)
sim.step_time(1.0)

print(line + "Deleting particles outside 50 mL region" + line)
sim.execute_command("delete_atoms region 1")

print(line + "Letting remaining particles settle" + line)
sim.step_time(1.0)


# Extract particle properties as NumPy arrays
time = sim.time()
radii = sim.radii()
positions = sim.positions()

print("\n\n" + line)
print(f"Simulation time: {time} s\nParticle positions:\n{positions}")
print(f"Number of particles: {positions.shape}")
print(f"Number of NaN particles: {np.isnan(positions).any(axis = 1).sum()}")
print(f"Number of deletion steps to remove particles above 50 ml: {i}")
print(line + "\n\n")


# Save results as efficient binary NPY-formatted files
np.save(f"results/radii.npy", radii)
np.save(f"results/positions.npy", positions)

print("Finished! :D")
