#!/bin/bash

#SBATCH --job-name=oai
#SBATCH --output=../logs/oai.out
#SBATCH --error=../logs/oai.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80GB
#SBATCH --time=3-00:00:00
#SBATCH --gres=gpu:a100

# job info
train_file=main
model="ResNet101" # ResNet101, DenseNet121, AlexNet
dataset="oai" # chest, oai
num_classes=5 # 14 for chest, 5 for oai

optimizer="Adam" # Adam, SGD
batch_size=32
loss="FOCAL" #BCE, FOCAL
learning_rate=1e-4
momentum=0.9 # Specifically for SGD
weight_decay=1e-5
total_epoch=30

# start running
echo -e "GPUS = $CUDA_VISIBLE_DEVICES\n"
nvidia-smi

source /gpfs/scratch/zhangs18/Miniconda3/etc/profile.d/conda.sh
conda activate image

cd /gpfs/data/chopralab/$USER/MLRIC

python -m ${train_file} --model ${model} --dataset ${dataset} --num_classes ${num_classes} \
        --optimizer ${optimizer} --batch_size ${batch_size} --loss ${loss} --lr ${learning_rate} \
        --momentum ${momentum} --weight_decay ${weight_decay} --total_epoch ${total_epoch}