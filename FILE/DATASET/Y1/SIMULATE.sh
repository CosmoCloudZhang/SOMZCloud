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
#SBATCH -J DATASET_Y1_SIMULATE
#SBATCH --mail-user=YunHao.Zhang@ed.ac.uk

# Load modules
module load python
module load PrgEnv-gnu
module load cray-mpich/8.1.28
module load cray-hdf5-parallel

# Activate the conda environment
source $HOME/.bashrc
conda activate $RAILENV

# Initialize the process
TAG="Y1"
BASE_PATH="/pscratch/sd/y/yhzhang/ZCloud/"
BASE_FOLDER="/global/cfs/cdirs/lsst/groups/PZ/users/CosmoCloud/ZCloud/"

# Run the application
python -u "${BASE_PATH}FILE/DATASET/${TAG}/SIMULATE.py" --tag=$TAG --folder=$BASE_FOLDER