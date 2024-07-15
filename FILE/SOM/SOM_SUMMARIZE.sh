#!/bin/bash
#SBATCH -A m1727
#SBATCH --nodes=1
#SBATCH -q regular
#SBATCH -J SUMMARIZE
#SBATCH --ntasks=16
#SBATCH --time=24:00:00
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
LENGTH=16
BASE_PATH="/pscratch/sd/y/yhzhang/ZCloud/"
for INDEX in $(seq 1 $LENGTH); do
    # Set path of input variables
    NAME="SOM_SUMMARIZE${INDEX}"
    INPUT_PATH="${BASE_PATH}/DATA/SAMPLE/TEST_SAMPLE.hdf5"
    MODEL_PATH="${BASE_PATH}/DATA/SOM/SOM_INFORM${INDEX}.pkl"
    SPEC_PATH="${BASE_PATH}/DATA/SAMPLE/TRAIN_SAMPLE${INDEX}.hdf5"
    CONFIG_PATH="${BASE_PATH}/DATA/SOM/SOM_SUMMARIZE${INDEX}.yaml"
    # Set path of output variables
    SINGLE_PATH="${BASE_PATH}/DATA/SOM/SOM_SINGLE${INDEX}.hdf5"
    CELLID_PATH="${BASE_PATH}/DATA/SOM/SOM_CELLID${INDEX}.hdf5"
    OUTPUT_PATH="${BASE_PATH}/DATA/SOM/SOM_SUMMARIZE${INDEX}.hdf5"
    CLUSTER_PATH="${BASE_PATH}/DATA/SOM/SOM_CELL_FILE${INDEX}.hdf5"
    # Run applications
    python "${BASE_PATH}/FILE/SOM/SOM_SUMMARIZE.py" --path="${BASE_PATH}" --index=$INDEX &
    srun -u -N 1 -n 1 --cpus-per-task=$SLURM_CPUS_PER_TASK python3 -m ceci rail.estimation.algos.somoclu_som.SOMocluSummarizer --mpi --name=$NAME --input=$INPUT_PATH --model=$MODEL_PATH --spec_input=$SPEC_PATH --config=$CONFIG_PATH --single_NZ=$SINGLE_PATH --output=$OUTPUT_PATH --uncovered_cluster_file=$CLUSTER_PATH --cellid_output=$CELLID_PATH &
    # Control parallel execution
    if (( $INDEX % $SLURM_NTASKS == 0 )); then
        wait
    fi
done
wait