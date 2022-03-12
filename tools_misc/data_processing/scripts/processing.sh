#!/bin/bash 
#SBATCH --job-name=processing_wilds_job
#SBATCH --output=/scratch/users/stellasu/batch_job_out/data_processing/processing_wilds_job.out
#SBATCH --error=/scratch/users/stellasu/batch_job_out/data_processing/processing_wilds.err
#SBATCH --time=2-00:00:00
#SBATCH -p ibiis
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=8G
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=stellasu@stanford.edu

source ~/miniconda3/etc/profile.d/conda.sh
conda activate transformer-ssl

python3 /home/users/stellasu/data_processing/create_training_sets_by_data_split.py --input_image_dir=/scratch/users/stellasu/datasets/wilds/camelyon17_v1.0/ --dataset_label_path=/scratch/users/stellasu/datasets/wilds/camelyon17_v1.0/metadata_fnames_splits_modSplit.csv --dataset_output_dir=/scratch/users/stellasu/datasets/wilds/camelyon17_v1.0/ --image_format=png

#python3 /home/users/stellasu/data_processing/create_training_sets_by_data_split.py --input_image_dir=/scratch/users/stellasu/datasets/wilds/camelyon17_v1.0/ --dataset_label_path=/scratch/users/stellasu/datasets/wilds/camelyon17_v1.0/metadata_fnames_splits_modSplit.csv --dataset_output_dir=/scratch/groups/rubin/stellasu/wilds/camelyon17_v1.0/ --image_format=png
