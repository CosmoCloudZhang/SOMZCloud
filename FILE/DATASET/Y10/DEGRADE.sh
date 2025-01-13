#!/bin/bash
#SBATCH -A m1727
#SBATCH -q regular
#SBATCH --ntasks=1
#SBATCH --time=04:00:00
#SBATCH --mail-type=END
#SBATCH --constraint=cpu
#SBATCH -o LOG/%x_%j.out
#SBATCH --cpus-per-task=256
#SBATCH --ntasks-per-node=1
#SBATCH -J DATASET_Y1_DEGRADE
#SBATCH --mail-user=YunHao.Zhang@ed.ac.uk

# Load modules
module load python
module load PrgEnv-gnu
module load cray-mpich/8.1.28
module load cray-hdf5-parallel

# Activate the conda environment
source $HOME/.bashrc
conda activate $RAILENV

# Set environment
export NUMEXPR_MAX_THREADS=$SLURM_CPUS_PER_TASK
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export HDF5_USE_FILE_LOCKING=FALSE
export OMP_PROC_BIND=spread
export OMP_PLACES=threads

# Initialize the process
TAG="Y10"
NUMBER=500
BASE_PATH="/pscratch/sd/y/yhzhang/ZCloud/"
BASE_FOLDER="/global/cfs/cdirs/lsst/groups/PZ/users/CosmoCloud/ZCloud/"

# Run the application
python -u "${BASE_PATH}FILE/DATASET/${TAG}/DEGRADE.py" --tag=$TAG --number=$NUMBER --folder=$BASE_FOLDER