#!/bin/bash
#SBATCH -A m1727
#SBATCH --nodes=5
#SBATCH -q regular
#SBATCH --time=48:00:00
#SBATCH --mail-type=END
#SBATCH --constraint=cpu
#SBATCH -o LOG/%x_%j.out
#SBATCH --cpus-per-task=4
#SBATCH --ntasks-per-node=64
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
COUNT=5
NUMBER=400
BASE_PATH="/pscratch/sd/y/yhzhang/ZCloud/"
BASE_FOLDER="/global/cfs/cdirs/lsst/groups/PZ/users/yhzhang/ZCloud/"

for BIN in $(seq 1 $COUNT); do 
    for INDEX in $(seq 1 $NUMBER); do
        # Set path of input variables
        NAME="SUMMARIZE_LENS${INDEX}"
        INPUT_PATH="${BASE_FOLDER}/FZB/LENS/LENS${INDEX}/SAMPLE${BIN}.hdf5"
        CONFIG_PATH="${BASE_FOLDER}/FZB/LENS/LENS${INDEX}/SUMMARIZE${BIN}.yaml"
        # Set path of output variables
        SINGLE_PATH="${BASE_FOLDER}/FZB/LENS/LENS${INDEX}/SINGLE${BIN}.hdf5"
        OUTPUT_PATH="${BASE_FOLDER}/FZB/LENS/LENS${INDEX}/SUMMARIZE${BIN}.hdf5"
        # Run applications
        python -u "${BASE_PATH}/FILE/FZB/SUMMARIZE_LENS.py" --bin=$BIN --index=$INDEX --folder=$BASE_FOLDER &
        srun -u -N 1 -n 1 --cpus-per-task=$SLURM_CPUS_PER_TASK python -m ceci rail.estimation.algos.naive_stack.NaiveStackSummarizer --mpi --name=$NAME --input=$INPUT_PATH --config=$CONFIG_PATH --single_NZ=$SINGLE_PATH --output=$OUTPUT_PATH &
        # Control parallel execution
        if (( $INDEX % $SLURM_NTASKS_PER_NODE == 0 )); then
            wait 
        fi
    done
    wait
done
wait