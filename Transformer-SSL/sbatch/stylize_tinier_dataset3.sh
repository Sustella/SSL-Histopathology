#!/bin/bash 
#SBATCH --job-name=stylize_tinier_dataset3
#SBATCH --output=stylize_tinier_dataset3.out
#SBATCH --error=stylize_tinier_dataset3.err
#SBATCH --time=2-00:00:00
#SBATCH -p owners
#SBATCH --gpus 1
#SBATCH --mem 16G
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=rikiya@stanford.edu

source ~/miniconda3/etc/profile.d/conda.sh
conda activate transformer-ssl

module load cuda/10.2.89

nvidia-smi
cd ~/contrastive_strap/SSL-Transformer-Histopathology/Transformer-SSL/style_transfer
python stylize_dataset.py --content-dir '/scratch/groups/rubin/rikiya/ssl_pretrain_tinier/train/0/' --style-dir '/scratch/users/rikiya/paintings/' --output-dir '/scratch/groups/rubin/rikiya/ssl_pretrain_tinier_stylized_rev/train/0/' --num-styles 8 --upto 32 --content-size 1024 --style-size 256
