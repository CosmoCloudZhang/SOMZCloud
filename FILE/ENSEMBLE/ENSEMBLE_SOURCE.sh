#!/bin/bash
#SBATCH -A m1727
#SBATCH --nodes=1
#SBATCH -q regular
#SBATCH --ntasks=1
#SBATCH --time=24:00:00
#SBATCH --mail-type=END
#SBATCH --constraint=cpu
#SBATCH -o LOG/%x_%a.out
#SBATCH -J ENSEMBLE_SOURCE
#SBATCH --cpus-per-task=256
#SBATCH --mail-user=YunHao.Zhang@ed.ac.uk

# Load modules
module load python
module load PrgEnv-gnu
module load cray-mpich/8.1.28
module load cray-hdf5-parallel

# Activate the conda environment
source $HOME/.bashrc
conda activate $CosmoENV

# Initialize the parallisation
SIZE=5
WIDTH=1000
LENGTH=400
BASE_PATH="/pscratch/sd/y/yhzhang/ZCloud/"
python -u $BASE_PATH/FILE/ENSEMBLE/ENSEMBLE_SOURCE.py --path="${BASE_PATH}" --size=$SIZE --width=$WIDTH --length=$LENGTH