#!/bin/bash
#SBATCH -A m1727
#SBATCH --nodes=1
#SBATCH -q regular
#SBATCH --ntasks=1
#SBATCH -J SOM_INFORM
#SBATCH --time=01:00:00
#SBATCH --mail-type=END
#SBATCH --constraint=cpu
#SBATCH -o LOG/%x_%a.out
#SBATCH --cpus-per-task=256
#SBATCH --mail-user=YunHao.Zhang@ed.ac.uk

# Load modules
module load python
module swap PrgEnv-${PE_ENV,,} PrgEnv-gnu

source $HOME/.bashrc
conda activate $RAILENV
module load cray-hdf5-parallel

# Set environment variables
export HDF5_USE_FILE_LOCKING=FALSE
export OMP_PROC_BIND=spread
export OMP_PLACES=threads

# Set path variables
NAME="SOM_INFORM"
BASE_PATH="/pscratch/sd/y/yhzhang/ZCloud/"
MODEL_PATH="${BASE_PATH}/DATA/SOM/SOM_INFORM.pkl"
CONFIG_PATH="${BASE_PATH}/DATA/SOM/SOM_INFORM.yaml"
INPUT_PATH="${BASE_PATH}/DATA/SAMPLE/TEST_SAMPLE.hdf5"
# Run applications
python "${BASE_PATH}/FILE/SOM/SOM_INFORM.py" --path="${BASE_PATH}" &
srun -u -N 1 -n 1 --cpus-per-task=$SLURM_CPUS_PER_TASK python3 -m ceci rail.estimation.algos.minisom_som.MiniSOMInformer --mpi  --name=$NAME --input=$INPUT_PATH --model=$MODEL_PATH --config=$CONFIG_PATH