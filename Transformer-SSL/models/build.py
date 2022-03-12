# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# Modified by Zhenda Xie
# --------------------------------------------------------

from functools import partial
from timm.models import vit_small_patch16_224
from timm.models import vit_tiny_patch16_224 
from timm.models import vision_transformer
from timm.models import resnet18, resnet50

from .swin_transformer import SwinTransformer
from .moby import MoBY

from torch import nn
from .vision_transformer import VisionTransformer

vit_models = dict(
    vit_tiny=vit_tiny_patch16_224,
    vit_small=vit_small_patch16_224,
)

cnn_models = dict(
    resnet50 = resnet50,
    resnet18 = resnet18,
)

def build_model(config):
    model_type = config.MODEL.TYPE
    encoder_type = config.MODEL.MOBY.ENCODER

    if encoder_type == 'swin':
        enc = partial(
            SwinTransformer,
            img_size=config.DATA.IMG_SIZE,
            patch_size=config.MODEL.SWIN.PATCH_SIZE,
            in_chans=config.MODEL.SWIN.IN_CHANS,
            embed_dim=config.MODEL.SWIN.EMBED_DIM,
            depths=config.MODEL.SWIN.DEPTHS,
            num_heads=config.MODEL.SWIN.NUM_HEADS,
            window_size=config.MODEL.SWIN.WINDOW_SIZE,
            mlp_ratio=config.MODEL.SWIN.MLP_RATIO,
            qkv_bias=config.MODEL.SWIN.QKV_BIAS,
            qk_scale=config.MODEL.SWIN.QK_SCALE,
            drop_rate=config.MODEL.DROP_RATE,
            ape=config.MODEL.SWIN.APE,
            patch_norm=config.MODEL.SWIN.PATCH_NORM,
            use_checkpoint=config.TRAIN.USE_CHECKPOINT,
            norm_befor_mlp=config.MODEL.SWIN.NORM_BEFORE_MLP,
        )
    
    elif encoder_type.startswith('vit'):
       enc = vit_models[encoder_type]
    
    #elif encoder_type.startswith('vit'):
    #     enc = partial(
    #        VisionTransformer,
    #        img_size=[config.DATA.IMG_SIZE], patch_size=16, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6),
    #    )

    elif encoder_type.startswith('resnet'):
        enc = cnn_models[encoder_type]

    else:
        raise NotImplementedError(f'--> Unknown encoder_type: {encoder_type}')

    if model_type == 'moby':
        encoder = enc(
            num_classes=0,
            drop_path_rate=config.MODEL.MOBY.ONLINE_DROP_PATH_RATE,
        )
        print('encoder.num_features {}'.format(encoder.num_features))
        encoder_k = enc(
            num_classes=0,
            drop_path_rate=config.MODEL.MOBY.TARGET_DROP_PATH_RATE,
        )
        model = MoBY(
            cfg=config,
            encoder=encoder,
            encoder_k=encoder_k,
            contrast_momentum=config.MODEL.MOBY.CONTRAST_MOMENTUM,
            contrast_temperature=config.MODEL.MOBY.CONTRAST_TEMPERATURE,
            contrast_num_negative=config.MODEL.MOBY.CONTRAST_NUM_NEGATIVE,
            proj_num_layers=config.MODEL.MOBY.PROJ_NUM_LAYERS,
            pred_num_layers=config.MODEL.MOBY.PRED_NUM_LAYERS,
        )
    elif model_type == 'linear':
        model = enc(
            num_classes=config.MODEL.NUM_CLASSES,
            drop_path_rate=config.MODEL.DROP_PATH_RATE,
        )
    else:
        raise NotImplementedError(f'--> Unknown model_type: {model_type}')

    return model
