#!/bin/bash
#SBATCH -A m1727
#SBATCH --nodes=1
#SBATCH -q regular
#SBATCH --time=24:00:00
#SBATCH --mail-type=END
#SBATCH --constraint=cpu
#SBATCH -o LOG/%x_%j.out
#SBATCH --cpus-per-task=8
#SBATCH -J CELL_Y1_NN_DATA
#SBATCH --ntasks-per-node=32
#SBATCH --mail-user=YunHao.Zhang@ed.ac.uk

# Load modules
module load conda
module load PrgEnv-gnu
module load cray-mpich/8.1.30
module load cray-hdf5-parallel

# Activate the conda environment
source $HOME/.bashrc
conda activate $CosmoENV

# Set environment
export NUMEXPR_MAX_THREADS=$SLURM_CPUS_PER_TASK
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export HDF5_USE_FILE_LOCKING=FALSE
export OMP_PROC_BIND=spread
export OMP_PLACES=threads

# Initialize the process
TAG="Y1"
NAME="NN"
BASE_PATH="/pscratch/sd/y/yhzhang/SOMZCloud/"
BASE_FOLDER="/global/cfs/cdirs/lsst/groups/MCP/CosmoCloud/SOMZCloud/"

# Run applications
LABEL_LIST=("ZERO" "HALF" "UNITY" "DOUBLE")
RANK_LIST=("DIR" "FIDUCIAL" "STACK" "PRODUCT" "TRUTH")

for LABEL in "${LABEL_LIST[@]}"; do
    for RANK in "${RANK_LIST[@]}"; do
        srun -u -N 1 -n 1 -c $SLURM_CPUS_PER_TASK python -u "${BASE_PATH}CELL/${TAG}/${NAME}/DATA.py" --tag=$TAG --name=$NAME --rank=$RANK --label=$LABEL --folder=$BASE_FOLDER &
    done
done
wait