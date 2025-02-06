#!/bin/bash
#SBATCH --job-name=eval
#SBATCH --nodes=1
#SBATCH --partition=align
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=2
#SBATCH --ntasks-per-node=1
#SBATCH --output=logs/%j.%x.info.log
#SBATCH --error=logs/%j.%x.error.log

srun python predict.py $1 $2 $3
srun python predict_cot.py $1 $2 $3
