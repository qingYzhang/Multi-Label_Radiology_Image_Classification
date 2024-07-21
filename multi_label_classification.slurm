#!/bin/bash

#SBATCH --job-name=multi_chest
#SBATCH --output=../logs/res_alex.out
#SBATCH --error=../logs/res_alex.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80GB
#SBATCH --time=3-00:00:00
#SBATCH --gres=gpu:a100

# job info
train_file=main_final
model="AlexNet" # ResNet101, DenseNet121, AlexNet

# start running
echo -e "GPUS = $CUDA_VISIBLE_DEVICES\n"
nvidia-smi

source /gpfs/scratch/zhangs18/Miniconda3/etc/profile.d/conda.sh
conda activate image

cd /gpfs/data/chopralab/$USER/MLRIC

python -m ${train_file} --model ${model}