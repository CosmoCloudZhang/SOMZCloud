#!/bin/bash
#SBATCH -A m1727
#SBATCH -J SELECT
#SBATCH --nodes=1
#SBATCH -q regular
#SBATCH --ntasks=16
#SBATCH --time=1:00:00
#SBATCH --mail-type=END
#SBATCH --constraint=cpu
#SBATCH -o LOG/%x_%a.out
#SBATCH --cpus-per-task=8
#SBATCH --mail-user=YunHao.Zhang@ed.ac.uk

# Load modules
module load python
module load PrgEnv-gnu
module load cray-mpich/8.1.28

# Set OpenMP environment
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OMP_PROC_Bind=spread
export OMP_PLACES=threads

# Activate the conda environment
source $HOME/.bashrc
conda activate $RAILENV

# Initialize the parallisation
LENGTH=16
for INDEX in $(seq 1 $LENGTH); do
    PATH_NAME="/pscratch/sd/y/yhzhang/ZCloud/"
    srun -u -N 1 -n 1 --cpus-per-task=$SLURM_CPUS_PER_TASK python3 "${PATH_NAME}/FILE/SELECT/SELECT.py" --path="${PATH_NAME}" --index="${INDEX}"
    # Control parallel execution
    if (( $INDEX % $SLURM_NTASKS == 0 )); then
        wait
    fi
done

wait