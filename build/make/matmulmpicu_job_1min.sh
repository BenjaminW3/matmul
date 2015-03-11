#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1           # use 1 GPU per node (i.e. use one GPU per task)
#SBATCH --exclusive
#SBATCH --nodes=4              # request nodes
#SBATCH --tasks-per-node=1     # allocate one task per node
#SBATCH --cpus-per-task=8      # use all 8 cores of one socket per task
#SBATCH --job-name=worpitz_matmulmpicu
#SBATCH --time=00:01:00

echo Starting Program

module load bullxmpi

srun ./matmulmpicu

echo Finished Program
