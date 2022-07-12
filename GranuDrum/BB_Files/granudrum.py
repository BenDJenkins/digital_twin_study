#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : granudrum.py
# License: GNU v3.0
# Author : Andrei Leonard Nicusan <a.l.nicusan@bham.ac.uk>
# Date   : 16.02.2022


import os
from collections import namedtuple

import numpy as np
from PIL import Image

import coexist
import konigcell as kc


# Create GranuDrum structure storing the system dimensions, assumed to be
# centred at (0, 0)
GranuDrum = namedtuple("GranuDrum", ["xlim", "ylim", "radius"])
granudrum = GranuDrum([-0.042, 0.042], [-0.042, 0.042], 0.042)

# Output image resolution
resolution = (500, 500)

# Relevant paths
script_path = "liggghts_script.sim"

print((
    "-----------------------------------------------------------------------\n"
    "[Python]\n"
    "Running GranuDrum simulation:\n"
    f"  Current Directory: {os.getcwd()}\n"
    f"  Simulation Script: {script_path}\n"
    "-----------------------------------------------------------------------\n"
), flush = True)


def encode_u8(image: np.ndarray, img_min = None, img_max = None) -> np.ndarray:
    '''Convert image from doubles to uint8 - i.e. encode real values to
    the [0-255] range.
    '''

    u8min = np.iinfo(np.uint8).min
    u8max = np.iinfo(np.uint8).max

    # If encoding values are given, copy and threshold image to avoid overflow
    if img_min is not None or img_max is not None:
        image = image.copy()

    if img_min is None:
        img_min = float(np.nanmin(image))
    else:
        image[image < img_min] = img_min

    if img_max is None:
        img_max = float(np.nanmax(image))
    else:
        image[image > img_max] = img_max

    img_bw = (image - img_min) / (img_max - img_min) * (u8max - u8min) + u8min
    img_bw = np.array(img_bw, dtype = np.uint8)

    return img_bw


def unroll_data(times, radii, positions, velocities):
    '''Concatenate particle data into individual trajectories, separated by
    rows of NaNs.
    '''
    num_timesteps = positions.shape[0]
    num_particles = positions.shape[1]

    positions = np.swapaxes(positions, 0, 1)    # (T, P, XYZ) -> (P, T, XYZ)
    positions = np.concatenate(positions)
    positions = np.insert(positions, np.s_[::num_timesteps], np.nan, axis=0)

    radii = np.swapaxes(radii, 0, 1)            # (T, P) -> (P, T)
    radii = np.concatenate(radii)
    radii = np.insert(radii, np.s_[::num_timesteps], np.nan)

    times = np.tile(times, num_particles)
    times = np.insert(times, np.s_[::num_timesteps], np.nan)

    velocities = np.swapaxes(velocities, 0, 1)  # (T, P, VXYZ) -> (P, T, VXYZ)
    velocities = np.concatenate(velocities)
    velocities = np.insert(velocities, np.s_[::num_timesteps], np.nan, axis=0)

    return times, radii, positions, velocities


def simulation_residence_distribution(
    granudrum: GranuDrum,
    image_shape: tuple,
    times: np.ndarray,
    radii: np.ndarray,
    positions: np.ndarray,
) -> kc.Pixels:
    '''Return the raw and uint8-binarised residence time distribution grid of
    the GranuDrum DEM simulation in the `konigcell.Pixels` format.
    '''

    # Extract GranuDrum dimensions
    xlim = granudrum.xlim
    ylim = granudrum.ylim

    # Compute time spent between consecutive particle positions
    dt = times[1:] - times[:-1]

    # Compute residence distribution in the XZ plane (i.e. granular drum side)
    sim_rtd = kc.dynamic2d(
        positions[:, [0, 2]],
        kc.RATIO,
        values = dt,
        radii = radii,
        resolution = image_shape,
        xlim = xlim,
        ylim = ylim,
        verbose = False,
    )

    # Save maximum RTD value encountered as a static variable attached to the
    # function itself
    if not hasattr(simulation_residence_distribution, "max_rtd_value"):
        simulation_residence_distribution.max_rtd_value = 0.

    simulation_residence_distribution.max_rtd_value = max(
        simulation_residence_distribution.max_rtd_value,
        np.nanmax(sim_rtd.pixels),
    )

    # Encode floating point pixel values to uint8
    sim_rtd2 = kc.Pixels.zeros(image_shape, xlim = xlim, ylim = ylim)
    maxval = simulation_residence_distribution.max_rtd_value
    sim_rtd2._pixels = encode_u8(sim_rtd.pixels, 0., maxval)

    # Return the original and binarised images
    return sim_rtd, sim_rtd2


def simulation_velocity_distribution(
    granudrum: GranuDrum,
    image_shape: tuple,
    velocities: np.ndarray,
    radii: np.ndarray,
    positions: np.ndarray,
) -> kc.Pixels:
    '''Return the raw and uint8-binarised velocity distribution grids (absolute
    and dimension-wise) of the GranuDrum DEM simulation in the
    `konigcell.Pixels` format.
    '''

    # Extract GranuDrum dimensions
    xlim = granudrum.xlim
    ylim = granudrum.ylim

    # Compute absolute velocities
    absvel = np.linalg.norm(velocities, axis = 1)

    def compute_grid(values):
        # Compute velocity distribution in the XZ plane - i.e. GranuDrum side
        return kc.dynamic_prob2d(
            positions[:, [0, 2]],
            values = values[:-1],
            radii = radii,
            resolution = image_shape,
            xlim = xlim,
            ylim = ylim,
            # max_workers = 1,
            verbose = False,
        )

    sim_vel = compute_grid(absvel)
    sim_vel2 = kc.Pixels.zeros(image_shape, xlim = xlim, ylim = ylim)
    sim_vel2._pixels = encode_u8(sim_vel.pixels)

    sim_velx = compute_grid(velocities[:, 0])
    sim_velx2 = kc.Pixels.zeros(image_shape, xlim = xlim, ylim = ylim)
    sim_velx2._pixels = encode_u8(sim_velx.pixels)

    sim_vely = compute_grid(velocities[:, 1])
    sim_vely2 = kc.Pixels.zeros(image_shape, xlim = xlim, ylim = ylim)
    sim_vely2._pixels = encode_u8(sim_vely.pixels)

    sim_velz = compute_grid(velocities[:, 2])
    sim_velz2 = kc.Pixels.zeros(image_shape, xlim = xlim, ylim = ylim)
    sim_velz2._pixels = encode_u8(sim_velz.pixels)

    # Return the original and binarised images
    return (
        sim_vel, sim_vel2,
        sim_velx, sim_velx2,
        sim_vely, sim_vely2,
        sim_velz, sim_velz2,
    )


def save_images(sample, times, radii, positions, velocities):
    # Unroll data into individual trajectories
    ut, ur, up, uv = unroll_data(times, radii, positions, velocities)

    pixels_rtd = simulation_residence_distribution(
        granudrum,
        resolution,
        ut, ur, up,
    )

    pixels_vel = simulation_velocity_distribution(
        granudrum,
        resolution,
        uv, ur, up,
    )

    # Save image as a compressed PNG photo; transpose and flip pixels to have
    # correct orientation for the PNG format

    i_str = str(sample)  # Add zeros in front of sample number in file name.
    sample_zeros = i_str.zfill(4)

    Image.fromarray(pixels_rtd[1].pixels.T[::-1]).save(
        f"results/gd_rtd_{sample_zeros}.png"
    )

    for i, v in enumerate(["abs", "x", "y", "z"]):
        Image.fromarray(pixels_vel[1 + 2 * i].pixels.T[::-1]).save(
            f"results/gd_vel{v}_{sample_zeros}.png"
        )


# Read in particle positions, formatted as (T, P, XYZ)
# times = np.load("results/times.npy")
# radii = np.load("results/radii.npy")
# positions = np.load("results/positions.npy")
# velocities = np.load("results/velocities.npy")


# Load modified simulation script
sim = coexist.LiggghtsSimulation(script_path, verbose=True)


# Simulation sampling frequency (Hz) and number of frames to use for one image
sampling = 100
frames = 20

tstart = 5
tend = 45
checkpoints = np.arange(tstart, tend, 1 / (sampling * frames))


# Save the complete data for the last 5 seconds of the simulation
last = 5
times_last = []
radii_last = []
positions_last = []
velocities_last = []


# Run simulation in batches of `frames` at a time
i = 0
while i < len(checkpoints):
    sample = i // frames
    line = "*" * 80
    print((
        f"\n\n{line}\n"
        f"Starting sample {sample} between times:\n"
        f"  {checkpoints[i]}s -\n"
        f"  {checkpoints[i + frames - 1]}s\n"
        f"{line}\n"
    ), flush = True)

    times = []
    radii = []
    positions = []
    velocities = []

    # Collect frames for current sample
    for j in range(frames):
        sim.step_to_time(checkpoints[i + j])

        times.append(sim.time())
        radii.append(sim.radii())
        positions.append(sim.positions())
        velocities.append(sim.velocities())

    # Stack lists of arrays
    times = np.array(times)
    radii = np.array(radii)
    positions = np.array(positions)
    velocities = np.array(velocities)

    # Save images of the RTD and velocity probability (abs, x, y, z)
    save_images(sample, times, radii, positions, velocities)

    # If we're in the last 5s of simulation, also save complete data
    if checkpoints[i] > tend - last:
        times_last.append(sim.time())
        radii_last.append(sim.radii())
        positions_last.append(sim.positions())
        velocities_last.append(sim.velocities())

    i += frames


# Save complete data from the last few seconds to disk
times_last = np.array(times_last)
radii_last = np.array(radii_last)
positions_last = np.array(positions_last)
velocities_last = np.array(velocities_last)

np.save(f"results/gd_times.npy", times_last)
np.save(f"results/gd_radii.npy", radii_last)
np.save(f"results/gd_positions.npy", positions_last)
np.save(f"results/gd_velocities.npy", velocities_last)

print("Finished! :D")
