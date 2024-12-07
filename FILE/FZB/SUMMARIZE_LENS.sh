#!/bin/bash
#SBATCH -A m1727
#SBATCH --nodes=5
#SBATCH -q regular
#SBATCH --time=04:00:00
#SBATCH --mail-type=END
#SBATCH --constraint=cpu
#SBATCH -o LOG/%x_%j.out
#SBATCH --cpus-per-task=8
#SBATCH --ntasks-per-node=32
#SBATCH -J FZB_SUMMARIZE_LENS
#SBATCH --mail-user=YunHao.Zhang@ed.ac.uk

# Load modules
module load python
module load PrgEnv-gnu
module load cray-mpich/8.1.28
module load cray-hdf5-parallel

# Activate the conda environment
source $HOME/.bashrc
conda activate $RAILENV

# Set OpenMP environment
export OMP_NUM_THREADS=16
export OMP_PLACES=threads
export OMP_PROC_Bind=spread

# Initialize the process
NUMBER=400
BASE_PATH="/pscratch/sd/y/yhzhang/ZCloud/"
BASE_FOLDER="/global/cfs/cdirs/lsst/groups/PZ/users/yhzhang/ZCloud/"

for INDEX in $(seq 1 $LENGTH); do
    # Set path of input variables
    NAME="SUMMARIZE_LENS${INDEX}"
    INPUT_PATH="${BASE_PATH}/DATA/SOM/LENS/LENS${INDEX}/SELECT${SLURM_ARRAY}.hdf5"
    CONFIG_PATH="${BASE_PATH}/DATA/SOM/LENS/LENS${INDEX}/SUMMARIZE${SLURM_ARRAY}.yaml"
    # Set path of output variables
    SINGLE_PATH="${BASE_PATH}/DATA/SOM/LENS/LENS${INDEX}/SINGLE${SLURM_ARRAY}.hdf5"
    OUTPUT_PATH="${BASE_PATH}/DATA/SOM/LENS/LENS${INDEX}/SUMMARIZE${SLURM_ARRAY}.hdf5"
    # Run applications
    python -u "${BASE_PATH}/FILE/FZB/SUMMARIZE_LENS.py" --path=$BASE_PATH --index=$INDEX &
    srun -u -N 1 -n 1 --cpus-per-task=$SLURM_CPUS_PER_TASK python -m ceci rail.estimation.algos.naive_stack.NaiveStackSummarizer --mpi --memmon --name=$NAME --input=$INPUT_PATH --config=$CONFIG_PATH --single_NZ=$SINGLE_PATH --output=$OUTPUT_PATH &
    # Control parallel execution
    if (( $INDEX % $SLURM_NTASKS == 0 )); then
        wait 
    fi
done
wait