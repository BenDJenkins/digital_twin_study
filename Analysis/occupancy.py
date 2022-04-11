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
import pickle

import numpy as np
import pandas as pd

import konigcell as kc
from PIL import Image


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


def load(path):
    with open(path, "rb") as f:
        obj = pickle.load(f)
    return obj


root = "/rds/projects/w/windowcr-granutools-engd/DigitalTwinStudy/Simulations"
parameterspath = f"{root}/Generated/parameters_unique.pickle"


parameters = load(parameterspath)

# Print unique parameter values
print("-" * 80 + "\nUnique Parameter Values:")
for k, v in parameters.items():
    print(f"{k}: {v}")
print("-" * 80)



if not os.path.isdir("images"):
    os.mkdir("images")


missing = []
simdirs = []

for i in [2, 1, 0]:
    p = parameters["psd"][i]

    for j in [2, 1, 0]:
        e = parameters["restitution"][j]

        for k in [6, 5, 4, 3, 2, 1, 0]:
            s = parameters["sliding"][k]

            for l in [6, 5, 4, 3, 2, 1, 0]:
                r = parameters["rolling"][l]

                for m in [6, 5, 4, 3, 2, 1, 0]:
                    c = parameters["cohesion"][m]

                    simdir = f"{root}/GranuBeaker/gb_{i}_{j}_{k}_{l}_{m}"
                    simdirs.append(simdir)

                    pospath = f"{simdir}/results/positions.npy"
                    radpath = f"{simdir}/results/radii.npy"

                    if not os.path.isfile(pospath) or not os.path.isfile(radpath):
                        missing.append(simdir)
                        continue

                    positions = np.load(pospath)
                    radii = np.load(radpath)
                    print(f"gb_{i}_{j}_{k}_{l}_{m} : {positions.shape}",
                          flush = True)

                    # Compute and save occupancy plot
                    pixels = kc.static2d(
                        positions[:, [0, 2]],
                        kc.INTERSECTION,
                        radii = radii,
                        resolution = (1000, 1000),
                        xlim = [-0.025, 0.025],
                        ylim = [0, 0.05],
                        verbose = False,
                    )

                    Image.fromarray(
                        encode_u8(
                            pixels.pixels.T[::-1],
                            img_max = 160e-9,
                        )
                    ).save(f"images/gb_{i}_{j}_{k}_{l}_{m}.png")

                    # Compute and save infinitesimal slice occupancy
                    p0 = np.array([0, 0, 0])
                    n0 = np.array([0, 1, 0])

                    d = np.sum((positions - p0) * n0, axis = 1) / np.linalg.norm(n0)
                    cond = np.abs(d) < radii

                    inf_positions = positions[cond]
                    inf_radii = np.sqrt(radii[cond]**2 - d[cond]**2)

                    pixels = kc.static2d(
                        inf_positions[:, [0, 2]],
                        kc.INTERSECTION,
                        radii = inf_radii,
                        resolution = (1000, 1000),
                        xlim = [-0.025, 0.025],
                        ylim = [0, 0.05],
                        verbose = False,
                    )

                    Image.fromarray(
                        encode_u8(
                            pixels.pixels.T[::-1],
                            img_max = np.prod(pixels.pixel_size),
                        )
                    ).save(f"images/slice_gb_{i}_{j}_{k}_{l}_{m}.png")


print(f"Collected {len(simdirs)} simulations; the following {len(missing)} are missing:")
[print(os.path.split(m)[1]) for m in missing]
