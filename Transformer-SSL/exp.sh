#!/bin/bash
#SBATCH --job-name=train_ssl_transformer
#SBATCH --output=$SCRATCH/batch_job_out/train_ssl_transformer_job.out
#SBATCH --error=$SCRATCH/batch_job_out/train_ssl_transformer.err
#SBATCH --partition=gpu
#SBATCH --time=1:00:00

### e.g. request 4 nodes with 1 gpu each, totally 4 gpus (WORLD_SIZE==4)
### Note: --gres=gpu:x should equal to ntasks-per-node
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH -C GPU_MEM:32GB
#SBATCH --cpus-per-task=1

### change 5-digit MASTER_PORT as you wish, slurm will raise Error if duplicated with others
### change WORLD_SIZE as gpus/node * num_nodes
export MASTER_PORT=12340
export WORLD_SIZE=4

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

### the command to run
srun python main.py --net resnet18 \
--lr 1e-3 --epochs 50 --other_args

