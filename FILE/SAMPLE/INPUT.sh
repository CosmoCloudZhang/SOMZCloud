#!/bin/bash

# Load modules
module load python
module load PrgEnv-gnu
module load cray-mpich/8.1.28

# Set OpenMP environment
export OMP_NUM_THREADS=1
export OMP_PLACES=threads
export OMP_PROC_Bind=spread

# Activate the conda environment
source $HOME/.bashrc
conda activate $CosmoENV

# Run the application
BASE_PATH="/pscratch/sd/y/yhzhang/ZCloud/"
BASE_FOLDER="/global/cfs/cdirs/lsst/groups/PZ/"
python -u "${BASE_PATH}/FILE/SAMPLE/INPUT.py" --path="${BASE_PATH}" --folder="${BASE_FOLDER}"