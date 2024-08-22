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

# Set path variables
NAME="SOM_INFORM"
BASE_PATH="/pscratch/sd/y/yhzhang/ZCloud/"
MODEL_PATH="${BASE_PATH}/DATA/SOM/SOM_INFORM.pkl"
INPUT_PATH="${BASE_PATH}/DATA/SOM/SOM_SAMPLE.hdf5"
CONFIG_PATH="${BASE_PATH}/DATA/SOM/SOM_INFORM.yaml"
# Run applications
python -u "${BASE_PATH}/FILE/SOM/SOM_INFORM.py" --path=$BASE_PATH &
srun -u -N 1 -n 1 --cpus-per-task=$SLURM_CPUS_PER_TASK python3 -m ceci rail.estimation.algos.somoclu_som.SOMocluInformer --mpi  --name=$NAME --input=$INPUT_PATH --model=$MODEL_PATH --config=$CONFIG_PATH