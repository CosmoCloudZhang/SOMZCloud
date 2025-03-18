#!/bin/bash
#SBATCH -A m1727
#SBATCH --nodes=1
#SBATCH -q regular
#SBATCH --time=48:00:00
#SBATCH --mail-type=END
#SBATCH --constraint=cpu
#SBATCH -o LOG/%x_%j.out
#SBATCH --cpus-per-task=8
#SBATCH -J CELL_Y10_NN_SHIFT
#SBATCH --ntasks-per-node=32
#SBATCH --mail-user=YunHao.Zhang@ed.ac.uk

# Load modules
module load python
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
TAG="Y10"
NAME="NN"
BASE_PATH="/pscratch/sd/y/yhzhang/ZCloud/"
BASE_FOLDER="/global/cfs/cdirs/lsst/groups/MCP/CosmoCloud/ZCloud/"

# Run applications
LABEL_LIST=("ZERO" "HALF" "UNITY" "DOUBLE")
TYPE_LIST=("SOM" "MODEL" "PRODUCT" "FIDUCIAL" "HISTOGRAM")

for LABEL in "${LABEL_LIST[@]}"; do
    for TYPE in "${TYPE_LIST[@]}"; do
        srun -u -N 1 -n 1 -c $SLURM_CPUS_PER_TASK python -u "${BASE_PATH}FILE/CELL/${TAG}/${NAME}/SHIFT.py" --tag=$TAG --name=$NAME --type=$TYPE --label=$LABEL --folder=$BASE_FOLDER &
    done
done
wait