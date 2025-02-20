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
module load python
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
TAG="Y1"
BASE_PATH="/pscratch/sd/y/yhzhang/ZCloud/"
BASE_FOLDER="/global/cfs/cdirs/lsst/groups/MCP/CosmoCloud/ZCloud/"

# Set variables
NAME="INFORM"
MODEL_NAME="${BASE_FOLDER}DATASET/${TAG}/SOM/INFORM.pkl"
INPUT_NAME="${BASE_FOLDER}DATASET/${TAG}/SOM/INFORM.hdf5"
CONFIG_NAME="${BASE_FOLDER}DATASET/${TAG}/SOM/INFORM.yaml"

# Run applications
python -u "${BASE_PATH}FILE/DATASET/${TAG}/SOM.py" --tag=$TAG --folder=$BASE_FOLDER &&
srun -u -N 1 -n $SLURM_NTASKS_PER_NODE -c $SLURM_CPUS_PER_TASK --cpu_bind=cores python -m ceci rail.estimation.algos.somoclu_som.SOMocluInformer --mpi --name=$NAME --input=$INPUT_NAME --model=$MODEL_NAME --config=$CONFIG_NAME & 
wait