#!/bin/bash 
#SBATCH --job-name=test_4gpu_owners_vlt_strap
#SBATCH --output=test_4gpu_owners_vlt_strap.out
#SBATCH --error=test_4gpu_owners_vlt_strap.err
#SBATCH --time=2-00:00:00
#SBATCH -p owners
#SBATCH --gpus 4
#SBATCH --gpu_cmode=shared 
#SBATCH -C GPU_MEM:32GB
#SBATCH --mem 128G
#SBATCH --nodes 1
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=rikiya@stanford.edu

source ~/miniconda3/etc/profile.d/conda.sh
conda activate transformer-ssl

module load cuda/10.2.89

nvidia-smi
cd ~/contrastive_strap/Transformer-SSL/
# python -m torch.distributed.launch --nproc_per_node 4 --master_port 12345  moby_main.py --cfg configs/moby_swin_tiny.yaml --data-path /scratch/groups/rubin/rikiya/ssl_pretrain_tiny --batch-size 128 --opts TRAIN.EPOCHS 300 --output /scratch/users/rikiya/ssl_pretrain_test_owners_4vlt
python -m torch.distributed.launch --nproc_per_node 4 --master_port 12345  moby_main.py --cfg configs/moby_swin_tiny_256.yaml --data-path /scratch/groups/rubin/rikiya/ssl_pretrain_tiny --batch-size 128 --opts TRAIN.EPOCHS 300 DATA.DATASET 'wsi' --output /scratch/users/stellasu/ssl_swin
