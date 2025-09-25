#!/bin/bash
#SBATCH -A m1727
#SBATCH --nodes=4
#SBATCH -q regular
#SBATCH --time=08:00:00
#SBATCH --mail-type=END
#SBATCH --constraint=cpu
#SBATCH -o LOG/%x_%j.out
#SBATCH --cpus-per-task=2
#SBATCH --ntasks-per-node=128
#SBATCH -J SUMMARIZE_Y1_COPPER_INFORM
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
TAG="Y1"
NUMBER=500
NAME="COPPER"
BASE_PATH="/pscratch/sd/y/yhzhang/SOMZCloud/"
BASE_FOLDER="/global/cfs/cdirs/lsst/groups/MCP/CosmoCloud/SOMZCloud/"

# Loop
for INDEX in $(seq 0 $NUMBER); do
    # Set variables
    INPUT_NAME="INFORM${INDEX}"
    INPUT_DATA="${BASE_FOLDER}DATASET/${TAG}/COMBINATION/DATA${INDEX}.hdf5"
    INPUT_MODEL="${BASE_FOLDER}SUMMARIZE/${TAG}/${NAME}/INFORM/INFORM${INDEX}.pkl"
    INPUT_CONFIG="${BASE_FOLDER}SUMMARIZE/${TAG}/${NAME}/INFORM/INFORM${INDEX}.yaml"
    # Run applications
    srun -u -N 1 -n 1 -c $SLURM_CPUS_PER_TASK python -u "${BASE_PATH}SUMMARIZE/${TAG}/${NAME}/INFORM.py" --tag=$TAG --name=$NAME --index=$INDEX --folder=$BASE_FOLDER &&
    srun -u -N 1 -n 1 -c $SLURM_CPUS_PER_TASK python -u -m ceci rail.estimation.algos.somoclu_som.SOMocluInformer --mpi --name=$INPUT_NAME --input=$INPUT_DATA --model=$INPUT_MODEL --config=$INPUT_CONFIG & 
    # Control parallel execution
    if (( $INDEX % $SLURM_NTASKS == 0 )); then
        wait
    fi
done
wait