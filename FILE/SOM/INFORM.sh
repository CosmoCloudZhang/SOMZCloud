#!/bin/bash
#SBATCH -A m1727
#SBATCH --nodes=1
#SBATCH -q regular
#SBATCH --ntasks=1
#SBATCH -J SOM_INFORM
#SBATCH --time=48:00:00
#SBATCH --mail-type=END
#SBATCH --constraint=cpu
#SBATCH -o LOG/%x_%j.out
#SBATCH --cpus-per-task=256
#SBATCH --mail-user=YunHao.Zhang@ed.ac.uk

# Load modules
module load python
module load PrgEnv-gnu
module load cray-mpich/8.1.28
module load cray-hdf5-parallel

# Activate the conda environment
source $HOME/.bashrc
conda activate $RAILENV

# Set environment variables
export HDF5_USE_FILE_LOCKING=FALSE
export OMP_PROC_BIND=spread
export OMP_PLACES=threads

# Initialize the process
NUMBER=400
BASE_PATH="/pscratch/sd/y/yhzhang/ZCloud/"
BASE_FOLDER="/global/cfs/cdirs/lsst/groups/PZ/users/yhzhang/ZCloud/"

# Set variables
NAME="INFORM"
MODEL_PATH="${BASE_FOLDER}SOM/INFORM/INFORM.pkl"
INPUT_PATH="${BASE_FOLDER}SOM/INFORM/INFORM.hdf5"
CONFIG_PATH="${BASE_FOLDER}SOM/INFORM/INFORM.yaml"

# Run applications
python -u "${BASE_PATH}FILE/SOM/INFORM.py" --number=$NUMBER --folder=$BASE_FOLDER &
srun -u -N 1 -n 1 --cpus-per-task=$SLURM_CPUS_PER_TASK python3 -m ceci rail.estimation.algos.somoclu_som.SOMocluInformer --mpi --name=$NAME --input=$INPUT_PATH --model=$MODEL_PATH --config=$CONFIG_PATH & 
wait