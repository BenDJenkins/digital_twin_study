#!/bin/bash
#SBATCH --time 2-0:0
#SBATCH --ntasks 16
#SBATCH --qos bbdefault
#SBATCH --account=windowcr-granutools-engd
#SBATCH --mail-type ALL
#SBATCH --constraint cascadelake



set -e
module purge; module load bluebear

module load bear-apps/2020a
module load PICI-LIGGGHTS/20210424-foss-2020a-Python-3.8.2
module load SciPy-bundle/2020.03-foss-2020a-Python-3.8.2

# Create virtual environment for installing Coexist
export VENV_DIR="${HOME}/virtual-environments"
export VENV_PATH="${VENV_DIR}/coexist-${BB_CPU}"

# Create a master venv directory if necessary
mkdir -p ${VENV_DIR}

# Check if virtual environment exists and create it if not
if [[ ! -d ${VENV_PATH} ]]; then
    virtualenv --system-site-packages ${VENV_PATH}

    source ${VENV_PATH}/bin/activate

    echo "Created venv, going to install coexist"

    pip install /rds/homes/b/bdj746/Coexist-master/
else
    source ${VENV_PATH}/bin/activate
fi

python3 simulation_script.py
