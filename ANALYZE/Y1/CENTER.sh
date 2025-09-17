#!/bin/bash
#SBATCH -A m1727
#SBATCH --nodes=1
#SBATCH -q regular
#SBATCH --time=04:00:00
#SBATCH --mail-type=END
#SBATCH --constraint=cpu
#SBATCH -o LOG/%x_%j.out
#SBATCH --cpus-per-task=64
#SBATCH --ntasks-per-node=4
#SBATCH -J ANALYZE_Y1_CENTER
#SBATCH --mail-user=YunHao.Zhang@ed.ac.uk

# Load modules
module load conda
module load PrgEnv-gnu
module load cray-mpich/8.1.30
module load cray-hdf5-parallel

# Activate the conda environment
source $HOME/.bashrc
conda activate $RAILENV

# Initialize the process
TAG="Y1"
NUMBER=500
BASE_PATH="/pscratch/sd/y/yhzhang/SOMZCloud/"
BASE_FOLDER="/global/cfs/cdirs/lsst/groups/MCP/CosmoCloud/SOMZCloud/"

# Run the application
LABEL_LIST=("DIR"  "STACK" "HYBRID" "TRUTH")

for LABEL in "${LABEL_LIST[@]}"; do
    # Run the application
    python -u "${BASE_PATH}ANALYZE/${TAG}/CENTER.py" --tag=$TAG --label=$LABEL --folder=$BASE_FOLDER &
done
wait