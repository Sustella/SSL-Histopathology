#!/bin/bash 
#SBATCH --job-name=test_4gpu_owners_vlt_vit_stain_norm_tinier
#SBATCH --output=test_4gpu_owners_vlt_vit_stain_norm_tinier.out
#SBATCH --error=test_4gpu_owners_vlt_vit_stain_norm_tinier.err
#SBATCH --time=2-00:00:00
#SBATCH -p owners
#SBATCH --gpus 4
#SBATCH --gpu_cmode=shared 
#SBATCH -C GPU_MEM:32GB
#SBATCH --mem 64G
#SBATCH --nodes 1
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=rikiya@stanford.edu

source ~/miniconda3/etc/profile.d/conda.sh
conda activate transformer-ssl

module load cuda/10.2.89

nvidia-smi
cd ~/contrastive_strap/SSL-Transformer-Histopathology/Transformer-SSL/
python -m torch.distributed.launch --nproc_per_node 4 --master_port 12346  moby_main.py --cfg configs/moby_vit_tiny.yaml --data-path /scratch/groups/rubin/rikiya/ssl_pretrain_tinier --batch-size 128 --opts TRAIN.EPOCHS 300 DATA.DATASET 'wsi' AUG.TRANSFORMATION 'stain_norm' --output /scratch/users/rikiya/ssl_pretrain_test_owners_4vlt_vit_stain_norm_tinier
