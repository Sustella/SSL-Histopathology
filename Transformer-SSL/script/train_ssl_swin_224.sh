#!/bin/bash 
#SBATCH --job-name=train_ssl_swin_224_job
#SBATCH --output=/scratch/groups/rubin/stellasu/training_results/train_ssl_swin_224_job.out
#SBATCH --error=/scratch/groups/rubin/stellasu/training_results/train_ssl_swin_224.err
#SBATCH --time=2-00:00:00
#SBATCH -p gpu
#SBATCH -G 1
#SBATCH -C GPU_MEM:32GB
#SBATCH --cpus-per-task 1
#SBATCH --nodes=1
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=stellasu@stanford.edu

### init virtual environment if needed
source ~/miniconda3/etc/profile.d/conda.sh
conda activate transformer-ssl

python3 -m torch.distributed.launch --nproc_per_node 1 --master_port 39457  /home/users/stellasu/SSL-Transformer-Histopathology/Transformer-SSL/moby_main.py --cfg /home/users/stellasu/SSL-Transformer-Histopathology/Transformer-SSL/configs/moby_swin_tiny_224.yaml --data-path /scratch/groups/rubin/rikiya/ssl_pretrain --batch-size 64 --opts TRAIN.EPOCHS 300 DATA.DATASET 'wsi' --local_rank 0 --output /scratch/users/stellasu/ssl_swin_224


