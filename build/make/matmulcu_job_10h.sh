#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1           # use 1 GPU per node (i.e. use one GPU per task)
#SBATCH --exclusive
#SBATCH --nodes=1              # request 1 node
#SBATCH --tasks-per-node=1     # allocate one task per node
#SBATCH --cpus-per-task=8      # use all 8 cores of one socket per task
#SBATCH --job-name=worpitz_matmul_cuda
#SBATCH --time=10:00:00

echo Starting Program

srun ./matmulcu

echo Finished Program
