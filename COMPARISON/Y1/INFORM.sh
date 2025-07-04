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
#SBATCH -J COMPARISON_Y1_INFORM
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
NUMBER=500
BASE_PATH="/pscratch/sd/y/yhzhang/SOMZCloud/"
BASE_FOLDER="/global/cfs/cdirs/lsst/groups/MCP/CosmoCloud/SOMZCloud/"

for INDEX in $(seq 0 $NUMBER); do
    # Set variables
    NAME="INFORM${INDEX}"
    INPUT_PATH="${BASE_FOLDER}DATASET/${TAG}/DEGRADATION/DATA${INDEX}.hdf5"
    MODEL_PATH="${BASE_FOLDER}COMPARISON/${TAG}/INFORM/INFORM${INDEX}.pkl"
    CONFIG_PATH="${BASE_FOLDER}COMPARISON/${TAG}/INFORM/INFORM${INDEX}.yaml"
    # Run applications
    python -u "${BASE_PATH}COMPARISON/${TAG}/INFORM.py" --tag=$TAG --index=$INDEX --folder=$BASE_FOLDER &&
    srun -u -N 1 -n 1 -c $SLURM_CPUS_PER_TASK python -m ceci rail.estimation.algos.flexzboost.FlexZBoostInformer --mpi --name=$NAME --input=$INPUT_PATH --model=$MODEL_PATH --config=$CONFIG_PATH &
    # COMPARISON parallel execution
    if (( $INDEX % $SLURM_NTASKS == 0 )); then
        wait
    fi
done
wait