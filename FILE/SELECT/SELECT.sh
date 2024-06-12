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
conda activate $RAILENV

# Run the application
INDEX=1
PATH_NAME="/pscratch/sd/y/yhzhang/ZCloud/"
python "${PATH_NAME}/FILE/SELECT/SELECT.py" --path="${PATH_NAME}" --index="${INDEX}"