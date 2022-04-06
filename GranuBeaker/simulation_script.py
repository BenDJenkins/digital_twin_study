#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : simulation_script.py
# License: GNU v3.0


import os

import numpy as np
import coexist


# Generate a new LIGGGHTS simulation from the `cylinder_template.sim` template
nparticles = 10  # 28294

sliding = 0.8
rolling = 0.7
restitution = 0.9
cohesion = 100000
pdd = 'large'
density = 1000

# Directory to save results to
results_dir = "results"
if not os.path.isdir(results_dir):
    os.mkdir(results_dir)


# Load simulation template lines as a list[str] and modify input parameters
with open(f"granubeaker_{pdd}.sim", "r") as f:
    sim_script = f.readlines()


# Simulation log path
sim_script[1] = f"log {results_dir}/granubeaker.log\n"
sim_script[10] = f"variable N equal {nparticles}\n"


# Parameter naming:
#    PP  = Particle-Particle
#    PW  = Particle-Wall (cylindrical hull of the granubeaker)
#    PSW = Particle-Sidewall (circular sides of the granubeaker)
sim_script[22] = f"variable fricPP equal {sliding}\n"
sim_script[23] = f"variable fricPW equal {sliding}\n"
sim_script[24] = f"variable fricPSW equal {sliding}\n"

sim_script[27] = f"variable fricRollPP equal {rolling}\n"
sim_script[28] = f"variable fricRollPW equal {rolling}\n"
sim_script[29] = f"variable fricRollPSW equal {rolling}\n"

sim_script[32] = f"variable corPP equal {restitution}\n"
sim_script[33] = f"variable corPW equal {restitution}\n"
sim_script[34] = f"variable corPSW equal {restitution}\n"

sim_script[37] = f"variable cohPP equal {cohesion}\n"
sim_script[38] = f"variable cohPW equal {cohesion}\n"
sim_script[39] = f"variable cohPSW equal {cohesion}\n"

sim_script[42] = f"variable dens equal {density}\n"


# Save the simulation template with the modified parameters
sim_path = f"{results_dir}/granubeaker.sim"
with open(sim_path, "w") as f:
    f.writelines(sim_script)


# Load modified simulation script
sim = coexist.LiggghtsSimulation(sim_path, verbose=True)


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
