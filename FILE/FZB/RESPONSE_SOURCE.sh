#!/bin/bash
#SBATCH -A m1727
#SBATCH --nodes=4
#SBATCH -q regular
#SBATCH --time=48:00:00
#SBATCH --mail-type=END
#SBATCH --constraint=cpu
#SBATCH -o LOG/%x_%j.out
#SBATCH --cpus-per-task=32
#SBATCH --ntasks-per-node=8
#SBATCH -J FZB_RESPONSE_SOURCE
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
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export HDF5_USE_FILE_LOCKING=FALSE
export OMP_PROC_BIND=spread
export OMP_PLACES=threads

# Initialize the process
NUMBER=400
BASE_PATH="/pscratch/sd/y/yhzhang/ZCloud/"
BASE_FOLDER="/global/cfs/cdirs/lsst/groups/PZ/users/CosmoCloud/ZCloud/"

# Run applications
for INDEX in $(seq 1 $NUMBER); do
    srun -u -N 1 -n 1 --cpu-bind=none python -u "${BASE_PATH}/FILE/FZB/RESPONSE_SOURCE.py" --index=$INDEX --folder=$BASE_FOLDER &
    # Control parallel execution
    if (( $INDEX % $SLURM_NTASKS == 0 )); then
        wait 
    fi
done
wait