#!/bin/bash 
#SBATCH --job-name=train_ssl_swin_job
#SBATCH --output=/scratch/users/stellasu/batch_job_out/train_ssl_swin_job.out
#SBATCH --error=/scratch/users/stellasu/batch_job_out/train_ssl_swin.err
#SBATCH --time=2-00:00:00
#SBATCH -p gpu
#SBATCH -G 1
#SBATCH -C GPU_BRD:TESLA
#SBATCH -C GPU_MEM:32GB
#SBATCH --cpus-per-task 1
#SBATCH --nodes=1
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=stellasu@stanford.edu

source ~/miniconda3/etc/profile.d/conda.sh
conda activate transformer-ssl

python3 -m torch.distributed.launch --nproc_per_node 1 --master_port 92346  /home/users/stellasu/Transformer-SSL/moby_main.py --cfg /home/users/stellasu/Transformer-SSL/configs/moby_swin_tiny_256.yaml --data-path /scratch/groups/rubin/rikiya/ssl_pretrain --batch-size 64 --opts TRAIN.EPOCHS 100 DATA.DATASET 'wsi' --local_rank 0 --output /scratch/users/stellasu/ssl_swin

