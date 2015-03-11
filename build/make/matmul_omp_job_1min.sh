#!/bin/bash
#SBATCH --exclusive
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --partition=sandy
#SBATCH --job-name=worpitz_matmul
#SBATCH --time=00:01:00

echo Starting Program

export OMP_NUM_THREADS=$SLURM_CPUS_ON_NODE

srun ./matmul

echo Finished Program
