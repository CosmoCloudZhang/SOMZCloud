#!/bin/bash
#SBATCH -A m1727
#SBATCH --nodes=4
#SBATCH -q regular
#SBATCH --time=04:00:00
#SBATCH --mail-type=END
#SBATCH --constraint=cpu
#SBATCH -o LOG/%x_%j.out
#SBATCH --cpus-per-task=2
#SBATCH --ntasks-per-node=128
#SBATCH -J CONSTRAIN_Y1_ESTIMATE
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
BASE_PATH="/pscratch/sd/y/yhzhang/SOMZCloud/"
BASE_FOLDER="/global/cfs/cdirs/lsst/groups/MCP/CosmoCloud/SOMZCloud/"

for INDEX in $(seq 1 $NUMBER); do
    # Set variables
    NAME="ESTIMATE${INDEX}"
    MODEL_PATH="${BASE_FOLDER}CONSTRAIN/${TAG}/INFORM/INFORM${INDEX}.pkl"
    INPUT_PATH="${BASE_FOLDER}DATASET/${TAG}/APPLICATION/DATA${INDEX}.hdf5"
    CONFIG_PATH="${BASE_FOLDER}CONSTRAIN/${TAG}/ESTIMATE/ESTIMATE${INDEX}.yaml"
    OUTPUT_PATH="${BASE_FOLDER}CONSTRAIN/${TAG}/ESTIMATE/ESTIMATE${INDEX}.hdf5"
    # Run applications
    python -u "${BASE_PATH}CONSTRAIN/${TAG}/ESTIMATE.py" --tag=$TAG --index=$INDEX --folder=$BASE_FOLDER &&
    srun -u -N 1 -n 1 -c $SLURM_CPUS_PER_TASK python -m ceci rail.estimation.algos.flexzboost.FlexZBoostEstimator --mpi --name=$NAME --input=$INPUT_PATH --model=$MODEL_PATH --config=$CONFIG_PATH --output=$OUTPUT_PATH &
    # Control parallel execution
    if (( $INDEX % $SLURM_NTASKS == 0 )); then
        wait
    fi
done
wait