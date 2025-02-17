#!/bin/bash
#SBATCH -A m1727
#SBATCH --nodes=4
#SBATCH -q regular
#SBATCH --time=04:00:00
#SBATCH --mail-type=END
#SBATCH --constraint=cpu
#SBATCH -o LOG/%x_%j.out
#SBATCH --cpus-per-task=8
#SBATCH --ntasks-per-node=32
#SBATCH -J SUMMARIZE_Y10_HISTOGRAM_SOURCE
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
BASE_FOLDER="/global/cfs/cdirs/lsst/groups/MCP/CosmoCloud/ZCloud/"

# Run applications
for INDEX in $(seq 1 $NUMBER); do
    srun -u -N 1 -n 1 -c $SLURM_CPUS_PER_TASK python -u "${BASE_PATH}FILE/SUMMARIZE/${TAG}/HISTOGRAM_SOURCE.py" --tag=$TAG --index=$INDEX --folder=$BASE_FOLDER & 
    # Control parallel execution
    if (( $INDEX % $SLURM_NTASKS == 0 )); then
        wait
    fi
done
wait