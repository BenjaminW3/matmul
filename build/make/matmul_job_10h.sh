#!/bin/bash
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=sandy
#SBATCH --job-name=worpitz_matmul
#SBATCH --time=10:00:00

echo Starting Program

srun ./matmul

echo Finished Program
