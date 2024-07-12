#!/bin/bash
#SBATCH -A m1727
#SBATCH --nodes=1
#SBATCH -J SOURCE
#SBATCH -q regular
#SBATCH --ntasks=16
#SBATCH --time=02:00:00
#SBATCH --mail-type=END
#SBATCH --constraint=cpu
#SBATCH -o LOG/%x_%a.out
#SBATCH --cpus-per-task=16
#SBATCH --mail-user=YunHao.Zhang@ed.ac.uk

# Load modules
module load python
module swap PrgEnv-${PE_ENV,,} PrgEnv-gnu

source $HOME/.bashrc
conda activate $RAILENV

module load PrgEnv-gnu
module load cray-hdf5-parallel

# Set environment
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export HDF5_USE_FILE_LOCKING=FALSE
export OMP_PROC_BIND=spread
export OMP_PLACES=threads

# Initialize the parallisation
WIDTH=5
LENGTH=400
BASE_PATH="/pscratch/sd/y/yhzhang/ZCloud/"

for BIN in $(seq 0 $((WIDTH - 1))); do
    for INDEX in $(seq 1 $LENGTH); do
        # Set path variables
        NAME="SOURCE_SUMMARISE${INDEX}"
        CONFIG_PATH="${BASE_PATH}/DATA/SOURCE/SOURCE${INDEX}/CONFIG.yaml"
        INPUT_PATH="${BASE_PATH}/DATA/SOURCE/SOURCE${INDEX}/SELECT_PDF${BIN}.hdf5"
        SINGLE_NZ_PATH="${BASE_PATH}/DATA/SOURCE/SOURCE${INDEX}/SELECT_BIN${BIN}.hdf5"
        OUTPUT_PATH="${BASE_PATH}/DATA/SOURCE/SOURCE${INDEX}/SELECT_SUMMARIZE${BIN}.hdf5"
        # Run applications
        python "${BASE_PATH}/FILE/SUMMARIZE/SOURCE.py" --path="${BASE_PATH}" --index=$INDEX &
        srun -u -N 1 -n 1 --cpus-per-task=$SLURM_CPUS_PER_TASK python3 -m ceci rail.estimation.algos.naive_stack.NaiveStackSummarizer --mpi --name=$NAME --input=$INPUT_PATH --config=$CONFIG_PATH --output=$OUTPUT_PATH --single_NZ=$SINGLE_NZ_PATH &
        # Control parallel execution
        if (( INDEX % SLURM_NTASKS == 0 )); then
            wait
        fi
    done
    wait
done
wait