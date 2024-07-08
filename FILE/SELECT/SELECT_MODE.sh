#!/bin/bash
#SBATCH -A m1727
#SBATCH -J SELECT
#SBATCH --nodes=1
#SBATCH -q regular
#SBATCH --ntasks=1
#SBATCH --time=24:00:00
#SBATCH --mail-type=END
#SBATCH --constraint=cpu
#SBATCH -o LOG/%x_%a.out
#SBATCH --cpus-per-task=256
#SBATCH --mail-user=YunHao.Zhang@ed.ac.uk

# Load modules
module load python
module load PrgEnv-gnu
module load cray-mpich/8.1.28

# Activate the conda environment
source $HOME/.bashrc
conda activate $RAILENV

# Set OpenMP environment
export OMP_NUM_THREADS=16
export OMP_PLACES=threads
export OMP_PROC_Bind=spread

# Initialize the parallisation
NUMBER=16
LENGTH=400
BASE_PATH="/pscratch/sd/y/yhzhang/ZCloud/"
srun -n 1 --cpu-bind=none python -u $BASE_PATH/FILE/SELECT/SELECT_MODE.py --path="${BASE_PATH}" --number=$NUMBER --length=$LENGTH &
wait