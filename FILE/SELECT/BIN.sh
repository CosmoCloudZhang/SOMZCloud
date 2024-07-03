#!/bin/bash
#SBATCH -J BIN
#SBATCH -A m1727
#SBATCH --nodes=1
#SBATCH -q regular
#SBATCH --ntasks=1
#SBATCH --time=01:00:00
#SBATCH --mail-type=END
#SBATCH --constraint=cpu
#SBATCH -o LOG/%x_%a.out
#SBATCH --cpus-per-task=128
#SBATCH --mail-user=YunHao.Zhang@ed.ac.uk

# Load modules
module load python
module load PrgEnv-gnu
module load cray-mpich/8.1.28

# Activate the conda environment
source $HOME/.bashrc
conda activate $CosmoENV

# Set OpenMP environment
export OMP_NUM_THREADS=64
export OMP_PLACES=threads
export OMP_PROC_Bind=spread

# Initialize the parallisation
BASE_PATH="/pscratch/sd/y/yhzhang/ZCloud/"
python -u $BASE_PATH/FILE/SELECT/BIN.py --path="${BASE_PATH}"
wait