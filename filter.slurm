#!/bin/bash

#SBATCH --job-name=filter_100000
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80GB
#SBATCH --time=3-00:00:00
#SBATCH --gres=gpu:a100

# job info
train_file=view_filter


# start running
echo -e "GPUS = $CUDA_VISIBLE_DEVICES\n"
nvidia-smi

source /gpfs/scratch/zhangs18/Miniconda3/etc/profile.d/conda.sh
conda activate image

cd /gpfs/data/chopralab/$USER/MLRIC

python -m ${train_file}