#!/bin/bash 
#SBATCH --job-name=stain_aug_batch1
#SBATCH --output=stain_aug_batch1.out
#SBATCH --error=stain_aug_batch1.err
#SBATCH --time=2-00:00:00
#SBATCH -p normal
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=32G
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=stellaktsu@gmail.com

source ~/miniconda3/etc/profile.d/conda.sh
conda activate transformer-ssl

cd ~/histopathology_augmentation
 
python stain_aug.py --batch_num=1 > stain_aug_scripts/batch1.out
