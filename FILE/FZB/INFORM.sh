#!/bin/bash
#SBATCH -A m1727
#SBATCH --nodes=1
#SBATCH -q regular
#SBATCH -J FZB_INFORM
#SBATCH --time=48:00:00
#SBATCH --mail-type=END
#SBATCH --constraint=cpu
#SBATCH -o LOG/%x_%j.out
#SBATCH --cpus-per-task=8
#SBATCH --ntasks-per-node=32
#SBATCH --mail-user=YunHao.Zhang@ed.ac.uk

# Load modules
module load python
module load PrgEnv-gnu
module load cray-mpich/8.1.28
module load cray-hdf5-parallel

# Activate the conda environment
source $HOME/.bashrc
conda activate $RAILENV

# Set environment
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export HDF5_USE_FILE_LOCKING=FALSE
export OMP_PROC_BIND=spread
export OMP_PLACES=threads

# Initialize the process
NUMBER=400
BASE_PATH="/pscratch/sd/y/yhzhang/ZCloud/"
BASE_FOLDER="/global/cfs/cdirs/lsst/groups/PZ/users/yhzhang/ZCloud/"

for INDEX in $(seq 1 $NUMBER); do
    # Set variables
    NAME="INFORM${INDEX}"
    MODEL_PATH="${BASE_FOLDER}FZB/INFORM/INFORM${INDEX}.pkl"
    CONFIG_PATH="${BASE_FOLDER}FZB/INFORM/INFORM${INDEX}.yaml"
    INPUT_PATH="${BASE_FOLDER}DATASET/COMBINATION/DATA${INDEX}.hdf5"
    # Run applications
    python -u "${BASE_PATH}FILE/FZB/INFORM.py" --index=$INDEX --folder=$BASE_FOLDER &
    srun -u -N 1 -n 1 --cpus-per-task=$SLURM_CPUS_PER_TASK python -m ceci rail.estimation.algos.flexzboost.FlexZBoostInformer --mpi --name=$NAME --input=$INPUT_PATH --model=$MODEL_PATH --config=$CONFIG_PATH &
    # Control parallel execution
    if (( $INDEX % $SLURM_NTASKS == 0 )); then
        wait
    fi
done
wait