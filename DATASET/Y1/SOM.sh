#!/bin/bash
#SBATCH -A m1727
#SBATCH --nodes=1
#SBATCH -q regular
#SBATCH --time=24:00:00
#SBATCH --mail-type=END
#SBATCH --constraint=cpu
#SBATCH -o LOG/%x_%j.out
#SBATCH -J DATASET_Y1_SOM
#SBATCH --cpus-per-task=256
#SBATCH --ntasks-per-node=1
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
BASE_PATH="/pscratch/sd/y/yhzhang/SOMZCloud/"
BASE_FOLDER="/global/cfs/cdirs/lsst/groups/MCP/CosmoCloud/SOMZCloud/"

# Set variables
INPUT_NAME="INFORM"
INPUT_MODEL="${BASE_FOLDER}DATASET/${TAG}/SOM/INFORM.pkl"
INPUT_DATA="${BASE_FOLDER}DATASET/${TAG}/SOM/INFORM.hdf5"
INPUT_CONFIG="${BASE_FOLDER}DATASET/${TAG}/SOM/INFORM.yaml"

# Run applications
python -u "${BASE_PATH}DATASET/${TAG}/SOM.py" --tag=$TAG --folder=$BASE_FOLDER &&
python -m ceci rail.estimation.algos.somoclu_som.SOMocluInformer --mpi --name=$INPUT_NAME --input=$INPUT_DATA --model=$INPUT_MODEL --config=$INPUT_CONFIG & 
wait