#!/bin/bash
#SBATCH -A m1727
#SBATCH --nodes=1
#SBATCH -q regular
#SBATCH --ntasks=16
#SBATCH --time=24:00:00
#SBATCH --mail-type=END
#SBATCH --constraint=cpu
#SBATCH -o LOG/%x_%a.out
#SBATCH --cpus-per-task=16
#SBATCH -J FZB_SUMMARIZE_SOURCE
#SBATCH --mail-user=YunHao.Zhang@ed.ac.uk

# Load modules
module load python
module swap PrgEnv-${PE_ENV,,} PrgEnv-gnu

source $HOME/.bashrc
conda activate $RAILENV
module load cray-hdf5-parallel

# Set environment
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export HDF5_USE_FILE_LOCKING=FALSE
export OMP_PROC_BIND=spread
export OMP_PLACES=threads

# Initialize the parallisation
WIDTH=5
LENGTH=16
BASE_PATH="/pscratch/sd/y/yhzhang/ZCloud/"
for BIN in $(seq 1 $WIDTH); do
    for INDEX in $(seq 1 $LENGTH); do
        # Set path of input variables
        NAME="FZB_SUMMARIZE_SOURCE${INDEX}"
        CONFIG_PATH="${BASE_PATH}/DATA/FZB/SOURCE/SOURCE${INDEX}/FZB_SUMMARIZE.yaml"
        INPUT_PATH="${BASE_PATH}/DATA/FZB/SOURCE/SOURCE${INDEX}/FZB_SELECT${BIN}.hdf5"
        SINGLE_PATH="${BASE_PATH}/DATA/FZB/SOURCE/SOURCE${INDEX}/FZB_SINGLE${BIN}.hdf5"
        OUTPUT_PATH="${BASE_PATH}/DATA/FZB/SOURCE/SOURCE${INDEX}/FZB_SUMMARIZE${BIN}.hdf5"
        # Run applications
        python "${BASE_PATH}/FILE/FZB/FZB_SUMMARIZE_SOURCE.py" --path="${BASE_PATH}" --index=$INDEX &
        srun -u -N 1 -n 1 --cpus-per-task=$SLURM_CPUS_PER_TASK python3 -m ceci rail.estimation.algos.naive_stack.NaiveStackSummarizer --mpi --name=$NAME --input=$INPUT_PATH --config=$CONFIG_PATH --single_NZ=$SINGLE_PATH --output=$OUTPUT_PATH &
        # Control parallel execution
        if (( $INDEX % $SLURM_NTASKS == 0 )); then
            wait 
            sleep 30
        fi
    done
    wait
    sleep 30
done
wait