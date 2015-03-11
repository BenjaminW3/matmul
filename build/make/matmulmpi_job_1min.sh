#!/bin/bash
#SBATCH --exclusive
#SBATCH --nodes=4						# request nodes
#SBATCH --tasks-per-node=16				# 16 tasks on each node
#SBATCH --cpus-per-task=1				# 1 cores per task
#SBATCH --partition=sandy
#SBATCH --job-name=worpitz_matmulmpi
#SBATCH --time=00:01:00

echo Starting Program

module load intel/2013-sp1 bullxmpi

srun ./matmulmpi

echo Finished Program
