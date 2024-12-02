#!/bin/bash
#SBATCH -A m1727
#SBATCH --nodes=1
#SBATCH -q regular
#SBATCH --ntasks=1
#SBATCH -J AUGMENT
#SBATCH --time=04:00:00
#SBATCH --mail-type=END
#SBATCH --constraint=cpu
#SBATCH -o LOG/%x_%j.out
#SBATCH --cpus-per-task=256
#SBATCH --mail-user=YunHao.Zhang@ed.ac.uk

# Load modules
module load python
module load PrgEnv-gnu
module load cray-mpich/8.1.28
module load cray-hdf5-parallel

# Activate the conda environment
source $HOME/.bashrc
conda activate $RAILENV

# Set OpenMP environment
export OMP_PLACES=threads
export OMP_NUM_THREADS=64
export OMP_PROC_Bind=spread

# Initialize the parallisation
NUMBER=400
BASE_PATH="/pscratch/sd/y/yhzhang/ZCloud/"
BASE_FOLDER="/global/cfs/cdirs/lsst/groups/PZ/users/yhzhang/ZCloud/"
BASE_DIRECTORY="/global/cfs/cdirs/lsst/projecta/lsst/groups/CS/roman_rubin_2023_v1.1.3/"

# Run the application
python -u "${BASE_PATH}/FILE/DATASET/AUGMENT.py" --number=$NUMBER --folder=$BASE_FOLDER --directory=$BASE_DIRECTORY