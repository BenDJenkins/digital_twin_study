#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : simulation_script_template2.py
# License: GNU v3.0


import os
import numpy as np
import coexist
import konigcell as kc
import plotly.graph_objs as go
from PIL import Image as im


# Generate a new LIGGGHTS simulation from the `granudrum_template.sim` template
rpm = 30
nparticles = 15000

sliding = 0.01
rolling = 0
restitution = 0.5
cohesion = 1000
density = 1000
pdd = 'large'


# Directory to save results to
results_dir = "results"
if not os.path.isdir(results_dir):
    os.mkdir(results_dir)


# Load simulation template lines as a list[str] and modify input parameters
with open("granudrum_template.sim", "r") as f:
    sim_script = f.readlines()


# Simulation log path
sim_script[1] = f"log {results_dir}/granudrum.log\n"

sim_script[9] = f"variable rotationPeriod equal 60/{rpm}\n"
sim_script[10] = f"variable N equal {nparticles}\n"


# Parameter naming:
#    PP  = Particle-Particle
#    PW  = Particle-Wall (cylindrical hull of the GranuDrum)
#    PSW = Particle-Sidewall (circular sides of the GranuDrum)
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
sim_path = f"{results_dir}/granudrum.sim"
with open(sim_path, "w") as f:
    f.writelines(sim_script)


# Load modified simulation script
sim = coexist.LiggghtsSimulation(sim_path, verbose=True)

# Run simulation up to given time (s)
sim.step_to_time(0.1)

# Extract particle properties as NumPy arrays
times = sim.time()
radii = sim.radii()
positions = sim.positions()
velocities = sim.velocities()

# Record particle properties at 120 Hz from t = 0 s up to t = 40 s
checkpoints = np.arange(0.1, 40.1, 1 / 120)

times = []
radii = []
positions = []
velocities = []

i = 0

if not os.path.exists("plots"):    # Make plots directory if it doesn't exist already
    os.mkdir("plots")

for t in checkpoints:
    sim.step_to_time(t)

    positions_plot = sim.positions()
    radii_plot = sim.radii()

    positions_plot = positions_plot[np.newaxis, :, :]
    radii_plot = radii_plot[np.newaxis, :]

    positions2d = positions_plot[:, :, [0, 2]]
    positions2d = np.concatenate(positions2d, axis=0)
    radii2d = np.concatenate(radii_plot, axis=0)

    occupancy = kc.static2d(
        positions2d,
        kc.INTERSECTION,
        radii=radii2d,
        resolution=(500, 500),
        # xlim=[xmin, xmax],
        # ylim=[ymin, ymax],
    )

    # Plot occupancy grid as a heatmap - i.e. greyscale image
    fig = go.Figure()
    fig.add_trace(occupancy.heatmap_trace())

    fig.update_layout(
        title="Occupancy Grid",
        xaxis_title="x (mm)",
        yaxis_title="z (mm)",

        yaxis_scaleanchor="x",
        yaxis_scaleratio=1,

        template="plotly_white",
    )

    i_str = str(i)
    zerofilled_istr = i_str.zfill(6)

    fig.write_image(f"plots/gd_{zerofilled_istr}.png")
    im.open(f"plots/gd_{zerofilled_istr}.png").save(f"plots/gd_{zerofilled_istr}.bmp")
    os.remove(f"plots/gd_{zerofilled_istr}.png")

    i = i+1

    if t >= 35.1:

        times.append(sim.time())
        radii.append(sim.radii())
        positions.append(sim.positions())
        velocities.append(sim.velocities())


# Save results as efficient binary NPY-formatted files
np.save(f"{results_dir}/times.npy", times)
np.save(f"{results_dir}/radii.npy", radii)
np.save(f"{results_dir}/positions.npy", positions)
np.save(f"{results_dir}/velocities.npy", velocities)
