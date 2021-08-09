#!/bin/bash 
#SBATCH --job-name=stylize_tiny_dataset0
#SBATCH --output=stylize_tiny_dataset0.out
#SBATCH --error=stylize_tiny_dataset0.err
#SBATCH --time=7-00:00:00
#SBATCH -p ibiis
#SBATCH --gpus 1
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=rikiya@stanford.edu

source ~/miniconda3/etc/profile.d/conda.sh
conda activate transformer-ssl

module load cuda/10.2.89

nvidia-smi
cd ~/contrastive_strap/SSL-Transformer-Histopathology/Transformer-SSL/style_transfer
python stylize_dataset.py --content-dir '/scratch/groups/rubin/rikiya/ssl_pretrain_tiny/train/0/' --style-dir '/scratch/users/rikiya/paintings/' --output-dir '/scratch/groups/rubin/rikiya/ssl_pretrain_tiny_stylized/train/0/' --num-styles 4 --content-size 1024 --style-size 256
