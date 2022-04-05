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

# Inserting Particles
print(line + "Pouring particles" + line)
sim.step_time(2.0)

# Allowing particles to settle
print(line + "Letting remaining particles fall and settle" + line)
sim.step_time(1.0)

# First deletion step
radii_before_deletion = sim.radii()

print(line + f"Deleting particles outside 50 ml region. Round: {1}" + line)
sim.execute_command("delete_atoms region 1")
radii_after_deletion = sim.radii()
number_par_deleted = len(radii_before_deletion) - len(radii_after_deletion)

# Delete particles until none are deleted anymore
i = 1

while number_par_deleted > 0:

    radii_before_deletion = sim.radii()

    print(line + f"Deleting particles outside 50 ml region. Round: {i+1}" + line)
    sim.execute_command("delete_atoms region 1")
    print(line + "Letting remaining particles settle" + line)
    sim.step_time(1.0)

    radii_after_deletion = sim.radii()
    number_par_deleted = len(radii_before_deletion) - len(radii_after_deletion)

    radii_before_deletion = []
    radii_after_deletion = []

    i = 1 + i

# Extract particle properties as NumPy arrays
time = sim.time()
radii = sim.radii()
positions = sim.positions()

print("\n\n" + line)
print(f"Simulation time: {time} s\nNumber of Particles:{len(positions)} \nNumber of Deletions: {i}")
print(line + "\n\n")


# Save results as efficient binary NPY-formatted files
np.save(f"{results_dir}/radii.npy", radii)
np.save(f"{results_dir}/positions.npy", positions)
