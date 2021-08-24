#!/bin/bash 
#SBATCH --job-name=train_linear_swin_224_lr1_rikiya_job
#SBATCH --output=/scratch/users/stellasu/batch_job_out/linear_results/train_linear_swin_224_lr1_rikiya_job.out
#SBATCH --error=/scratch/users/stellasu/batch_job_out/linear_results/train_linear_swin_224_lr1_rikiya.err
#SBATCH --time=2-00:00:00
#SBATCH -p gpu
#SBATCH -G 1
#SBATCH -C GPU_MEM:32GB
#SBATCH --cpus-per-task 1
#SBATCH --nodes=1
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=stellasu@stanford.edu

### change 5-digit MASTER_PORT as you wish, slurm will raise Error if duplicated with others
### change WORLD_SIZE as gpus/node * num_nodes
export MASTER_PORT=59458
export WORLD_SIZE=1

### get the first node name as master address - customized for vgg slurm
### e.g. master(gnodee[2-5],gnoded1) == gnodee2
echo "NODELIST="${SLURM_NODELIST}

if [ ${SLURM_NODELIST:7:1} == "," ]; then
    echo "MASTER_ADDR="${SLURM_NODELIST:0:7}
    export MASTER_ADDR=${SLURM_NODELIST:0:7}
elif [ ${SLURM_NODELIST:6:1} == "[" ]; then
    echo "MASTER_ADDR="${SLURM_NODELIST:0:6}${SLURM_NODELIST:7:1}
    export MASTER_ADDR=${SLURM_NODELIST:0:6}${SLURM_NODELIST:7:1}
else
    echo "MASTER_ADDR="${SLURM_NODELIST}
    export MASTER_ADDR=${SLURM_NODELIST}
fi

### init virtual environment if needed
source ~/miniconda3/etc/profile.d/conda.sh
conda activate transformer-ssl

python3 -m torch.distributed.launch --nproc_per_node 1 --master_port 55458 $HOME/SSL-Transformer-Histopathology/Transformer-SSL/moby_linear.py --cfg $GROUP_SCRATCH/rikiya/checkpoints/configs/moby_swin_tiny.yaml --data-path /scratch/users/stellasu/datasets/wilds/camelyon17_v1.0/ --batch-size 128 --opts TRAIN.EPOCHS 80 TRAIN.WARMUP_EPOCHS 5  DATA.DATASET 'wsi' DATA.IMG_SIZE 224 --local_rank 0 --output $SCRATCH/ssl_checkpoints/ssl_pretrain_swin_baseline_tinier/ --num_classes 2 --lr 1.0

#python3 -m torch.distributed.launch --nproc_per_node 1 --master_port 89457 $HOME/SSL-Transformer-Histopathology/Transformer-SSL/moby_linear.py --cfg $HOME/SSL-Transformer-Histopathology/Transformer-SSL/configs/moby_swin_tiny_224.yaml --data-path /scratch/users/stellasu/datasets/wilds/camelyon17_v1.0/train_val_data --batch-size 256 --opts TRAIN.EPOCHS 20 TRAIN.WARMUP_EPOCHS 3  DATA.DATASET 'wsi' DATA.IMG_SIZE 224 --local_rank 0 --output /scratch/users/stellasu/ssl_swin_224 --num_classes 2
