#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : generate_granubeaker.py
# License: GNU v3.0
# Author : Andrei Leonard Nicusan <a.l.nicusan@bham.ac.uk>
# Date   : 16.02.2022


'''Generate GranuBeaker concrete scripts from the templates.
'''


import os
import pickle


root = "/rds/projects/w/windowcr-granutools-engd/DigitalTwinStudy/Simulations"
psds = ["small", "medium", "large"]


def modify_template(script, restitution, sliding, rolling, cohesion):

    script[22] = f"variable fricPP equal {sliding}\n"
    script[23] = f"variable fricPW equal {sliding}\n"
    script[24] = f"variable fricPSW equal {sliding}\n"

    script[27] = f"variable fricRollPP equal {rolling}\n"
    script[28] = f"variable fricRollPW equal {rolling}\n"
    script[29] = f"variable fricRollPSW equal {rolling}\n"

    script[32] = f"variable corPP equal {restitution}\n"
    script[33] = f"variable corPW equal {restitution}\n"
    script[34] = f"variable corPSW equal {restitution}\n"

    script[37] = f"variable cohPP equal {cohesion}\n"
    script[38] = f"variable cohPW equal {cohesion}\n"
    script[39] = f"variable cohPSW equal {cohesion}\n"

    return script


for i, psd in enumerate(psds):
    script_path = f"{root}/Templates/granubeaker_{psd}.sim"
    mesh_path = f"{root}/Templates/granubeaker_mesh"
    print("\n\n", script_path)

    # Read in LIGGGHTS script template
    with open(script_path) as f:
        script = f.readlines()

    # Read in parameters dictionary
    with open("parameters_unique.pickle", "rb") as f:
        parameters_unique = pickle.load(f)

    p = parameters_unique["psd"][i]
    for j, e in enumerate(parameters_unique["restitution"]):
        for k, s in enumerate(parameters_unique["sliding"]):
            for l, r in enumerate(parameters_unique["rolling"]):
                for m, c in enumerate(parameters_unique["cohesion"]):

                    # Create simulation directory
                    simdir = f"{root}/GranuBeaker/gb_{i}_{j}_{k}_{l}_{m}"
                    print(simdir)
                    if not os.path.isdir(simdir):
                        os.mkdir(simdir)

                    # Save current parameters
                    with open(f"{simdir}/parameters.txt", "w") as f:
                        f.write((
                            f"PSD:          {p}\n"
                            f"Restitution:  {e}\n"
                            f"Sliding:      {s}\n"
                            f"Rolling:      {r}\n"
                            f"Cohesion:     {c}\n"
                        ))

                    if not os.path.isdir(f"{simdir}/results"):
                        os.mkdir(f"{simdir}/results")

                    # Substitute parameter values and save
                    concrete = modify_template(
                        script,
                        e, s, r, c,
                    )

                    with open(f"{simdir}/liggghts_script.sim", "w") as f:
                        f.writelines(concrete)

                    # Copy mesh folder
                    os.system(f"cp -r {mesh_path} {simdir}/mesh")
