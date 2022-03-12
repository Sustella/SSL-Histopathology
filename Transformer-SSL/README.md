# Self-Supervised Contrastive Learning with Histopathology-Specific Augmentations

The following models and augmentations are supported:

Models

* MoBy with Resnet-50
* MoBy with Swin Transformers, a form of Vision Transformer

Augmentations

* Baseline: random crop + flip + gaussian blur
* Stain normalization + baseline
* Stain augmentation + baseline
* Style transfer (STRAP) + baseline

## Self-Supervised Training Commands

**MoBy with Resnet-50**

- Baseline

```python -m torch.distributed.launch --nproc_per_node 2 --master_port 32317  moby_main.py --cfg configs/moby_resnet50.yaml --data-path /scratch/groups/rubin/stellasu/ssl_pretrain_tiny_ --batch-size 128 --opts TRAIN.EPOCHS 100 TRAIN.BASE_LR 0.002 DATA.DATASET 'wsi' --output $SCRATCH/ssl_checkpoints_imagenetPretrain/resnet_baseline_tiny_lr0001_2gpu```

- Stain Normalization

```python -m torch.distributed.launch --nproc_per_node 2 --master_port 22320  moby_main.py --cfg configs/moby_resnet50.yaml --data-path /scratch/groups/rubin/stellasu/ssl_pretrain_tiny_ --batch-size 128 --opts AUG.TRANSFORMATION 'stain_norm' TRAIN.EPOCHS 100 TRAIN.BASE_LR 0.002 DATA.DATASET 'wsi' --output $SCRATCH/ssl_checkpoints_imagenetPretrain/resnet_stain_norm_tiny_lr0001_2gpu```

- Stain Augmentation

```python -m torch.distributed.launch --nproc_per_node 2 --master_port 12319  moby_main.py --cfg configs/moby_resnet50.yaml --data-path /scratch/groups/rubin/stellasu/ssl_pretrain_tiny_ --batch-size 128 --opts AUG.TRANSFORMATION 'stain_aug' TRAIN.EPOCHS 100 TRAIN.BASE_LR 0.002 DATA.DATASET 'wsi' --output $SCRATCH/ssl_checkpoints_imagenetPretrain/resnet_stain_aug_tiny_lr0001_2gpu```

- STRAP

```python -m torch.distributed.launch --nproc_per_node 2 --master_port 32319  moby_main.py --cfg configs/moby_resnet50.yaml --data-path /scratch/groups/rubin/stellasu/ssl_pretrain_tiny_ --batch-size 128 --opts AUG.TRANSFORMATION 'strap' TRAIN.EPOCHS 100 TRAIN.BASE_LR 0.002 DATA.DATASET 'wsi' --output $SCRATCH/ssl_checkpoints_imagenetPretrain/resnet_both_strap_tiny_lr0001_2gpu```


**MoBy with Swin Transformers**

- Baseline
`
```python -m torch.distributed.launch --nproc_per_node 2 --master_port 42147  moby_main.py --cfg configs/moby_swin_tiny.yaml --data-path /scratch/groups/rubin/stellasu/ssl_pretrain_tiny --batch-size 128 --opts TRAIN.EPOCHS 100 TRAIN.BASE_LR 0.002 DATA.DATASET 'wsi' --output $SCRATCH/ssl_checkpoints_imagenetPretrain/swin_baseline_tiny_lr0001_2gpu```

- Stain Normalization

```python -m torch.distributed.launch --nproc_per_node 2 --master_port 15342  moby_main.py --cfg configs/moby_swin_tiny.yaml --data-path /scratch/groups/rubin/stellasu/ssl_pretrain_tiny_ --batch-size 128 --opts AUG.TRANSFORMATION 'stain_norm' TRAIN.EPOCHS 100 TRAIN.BASE_LR 0.002 DATA.DATASET 'wsi' --output $SCRATCH/ssl_checkpoints_imagenetPretrain/swin_stain_norm_tiny_lr0001_2gpu```

- Stain Augmentation
 
```python -m torch.distributed.launch --nproc_per_node 2 --master_port 35342  moby_main.py --cfg configs/moby_swin_tiny.yaml --data-path /scratch/groups/rubin/stellasu/ssl_pretrain_tiny_ --batch-size 128 --opts AUG.TRANSFORMATION 'stain_aug' TRAIN.EPOCHS 100 TRAIN.BASE_LR 0.002 DATA.DATASET 'wsi' --output $SCRATCH/ssl_checkpoints_imagenetPretrain/swin_stain_aug_tiny_lr0001_2gpu```
 
- STRAP

```python -m torch.distributed.launch --nproc_per_node 2 --master_port 32342  moby_main.py --cfg configs/moby_swin_tiny.yaml --data-path /scratch/groups/rubin/stellasu/ssl_pretrain_tiny --batch-size 128 --opts AUG.TRANSFORMATION 'strap' TRAIN.EPOCHS 100 TRAIN.BASE_LR 0.002 DATA.DATASET 'wsi' --output $SCRATCH/ssl_checkpoints_imagenetPretrain/swin_strap_noGrayscale_tiny_lr0001_2gpu```

## Linear Evaluation Commands

- Training

```python -m torch.distributed.launch --nproc_per_node 4 --master_port 32343  moby_linear.py --cfg configs/moby_resnet50.yaml --data-path /home/ttian_google_com/datasets/wilds/camelyon17_v1.0 --batch-size 128 --opts TRAIN.EPOCHS 100 DATA.DATASET 'wsi' --output /home/ttian_google_com/training/ssl_checkpoints_imagenetPretrain/resnet_baseline_ssl_pretrain_tinier_lr001 --num_classes 2 --lr 0.01 > ~/Transformer-SSL/Training_scripts/ssl_imagenetPretrain_linear_eval/resnet_baseline_lr001_linear_eval_lr001.out```

- Evaluation

```python -m torch.distributed.launch --nproc_per_node 1 --master_port 32343  moby_linear.py --cfg configs/moby_resnet50.yaml --data-path /home/ttian_google_com/datasets/wilds/camelyon17_v1.0 --batch-size 128 --opts DATA.DATASET 'wsi' --local_rank 0 --output /home/ttian_google_com/training/ssl_checkpoints_imagenetPretrain/resnet_baseline_ssl_pretrain_tinier_lr001 --num_classes 2 --eval --eval_set test --lr 0.01 > ~/Transformer-SSL/Training_scripts/ssl_imagenetPretrain_linear_eval/resnet_baseline_lr001_linear_eval_lr001_eval.out```


