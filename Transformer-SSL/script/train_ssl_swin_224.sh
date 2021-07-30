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

### change 5-digit MASTER_PORT as you wish, slurm will raise Error if duplicated with others
### change WORLD_SIZE as gpus/node * num_nodes
export MASTER_PORT=89457
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

python3 -m torch.distributed.launch --nproc_per_node 1 --master_port 89457  /home/users/stellasu/Transformer-SSL/moby_main.py --cfg /home/users/stellasu/Transformer-SSL/configs/moby_swin_tiny_224.yaml --data-path /scratch/groups/rubin/rikiya/ssl_pretrain --batch-size 64 --opts TRAIN.EPOCHS 300 DATA.DATASET 'wsi' --local_rank 0 --output /scratch/users/stellasu/ssl_swin_224


