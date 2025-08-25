#!/bin/bash
#SBATCH -A m1727
#SBATCH --nodes=1
#SBATCH -q regular
#SBATCH --time=08:00:00
#SBATCH --mail-type=END
#SBATCH --constraint=cpu
#SBATCH -o LOG/%x_%j.out
#SBATCH --cpus-per-task=16
#SBATCH -J FIGURE_Y10_SAMPLE
#SBATCH --ntasks-per-node=16
#SBATCH --mail-user=YunHao.Zhang@ed.ac.uk

# Load modules
module load conda
module load PrgEnv-gnu
module load cray-mpich/8.1.30
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
BASE_PATH="/pscratch/sd/y/yhzhang/SOMZCloud/"
BASE_FOLDER="/global/cfs/cdirs/lsst/groups/MCP/CosmoCloud/SOMZCloud/"

# Run the application
for INDEX in $(seq 0 $NUMBER); do
    srun -N 1 -n 1 -c $SLURM_CPUS_PER_TASK --cpu_bind=cores python -u "${BASE_PATH}FIGURE/${TAG}/SAMPLE.py" --tag=$TAG --index=$INDEX --folder=$BASE_FOLDER & 
    # Control parallel execution
    if (( $INDEX % $SLURM_NTASKS == 0 )); then
        wait
    fi
done
wait