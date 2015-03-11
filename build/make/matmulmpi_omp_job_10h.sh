#!/bin/bash
#SBATCH --exclusive
#SBATCH --nodes=4						# request nodes
#SBATCH --tasks-per-node=1				# one task on each node
#SBATCH --cpus-per-task=16				# cores per task
#SBATCH --partition=sandy
#SBATCH --job-name=worpitz_matmulmpi
#SBATCH --time=10:00:00

echo Starting Program

module load intel/2013-sp1 bullxmpi

export OMP_NUM_THREADS=$SLURM_CPUS_ON_NODE

srun ./matmulmpi

echo Finished Program
