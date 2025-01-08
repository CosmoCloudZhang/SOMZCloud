#!/bin/bash
#SBATCH -A m1727
#SBATCH --nodes=4
#SBATCH -q regular
#SBATCH --time=48:00:00
#SBATCH --mail-type=END
#SBATCH --constraint=cpu
#SBATCH -o LOG/%x_%j.out
#SBATCH --cpus-per-task=8
#SBATCH --ntasks-per-node=32
#SBATCH -J SOM_SUMMARIZE_SOURCE
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

# Initialize process
SIZE=5
NUMBER=400
BASE_PATH="/pscratch/sd/y/yhzhang/ZCloud/"
BASE_FOLDER="/global/cfs/cdirs/lsst/groups/PZ/users/CosmoCloud/ZCloud/"

# Loop
for BIN in $(seq 1 $SIZE); do
    for INDEX in $(seq 1 $NUMBER); do
        # Set path of input variables
        NAME="SUMMARIZE_SOURCE${INDEX}"
        MODEL_PATH="${BASE_FOLDER}SOM/INFORM/INFORM.pkl"
        SPEC_PATH="${BASE_FOLDER}DATASET/COMBINATION/DATA${INDEX}.hdf5"
        INPUT_PATH="${BASE_FOLDER}SOM/SOURCE/SOURCE${INDEX}/SAMPLE${BIN}.hdf5"
        CONFIG_PATH="${BASE_FOLDER}SOM/SOURCE/SOURCE${INDEX}/SUMMARIZE${BIN}.yaml"
        # Set path of output variables
        CELL_PATH="${BASE_FOLDER}SOM/SOURCE/SOURCE${INDEX}/CELL${BIN}.hdf5"
        SINGLE_PATH="${BASE_FOLDER}SOM/SOURCE/SOURCE${INDEX}/SINGLE${BIN}.hdf5"
        CLUSTER_PATH="${BASE_FOLDER}SOM/SOURCE/SOURCE${INDEX}/CLUSTER${BIN}.hdf5"
        OUTPUT_PATH="${BASE_FOLDER}SOM/SOURCE/SOURCE${INDEX}/SUMMARIZE${BIN}.hdf5"
        # Run applications
        python -u "${BASE_PATH}/FILE/SOM/SUMMARIZE_SOURCE.py" --bin=$BIN --index=$INDEX --folder=$BASE_FOLDER &
        srun -u -N 1 -n 1 --cpus-per-task=$SLURM_CPUS_PER_TASK python3 -m ceci rail.estimation.algos.somoclu_som.SOMocluSummarizer --mpi --name=$NAME --input=$INPUT_PATH --model=$MODEL_PATH --spec_input=$SPEC_PATH --config=$CONFIG_PATH --single_NZ=$SINGLE_PATH --output=$OUTPUT_PATH --uncovered_cluster_file=$CLUSTER_PATH --cellid_output=$CELL_PATH &
        # Control parallel execution
        if (( $INDEX % $SLURM_NTASKS == 0 )); then
            wait 
        fi
    done
    wait
done
wait