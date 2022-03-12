# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# Modified by Zhenda Xie
# --------------------------------------------------------

import os
import torch
import numpy as np
from PIL import Image, ImageFilter, ImageOps
import torch.distributed as dist
from torchvision import datasets, transforms
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import Mixup
from timm.data import create_transform
from timm.data.transforms import _pil_interp

from .cached_image_folder import CachedImageFolder
from .custom_image_folder import CustomImageFolder
from .samplers import SubsetRandomSampler
# from style_transfer import stylize
from histomicstk.preprocessing.augmentation.color_augmentation import rgb_perturb_stain_concentration
import staintools

def build_loader(config):
    config.defrost()
    dataset_train, config.MODEL.NUM_CLASSES = build_dataset(is_train=True, config=config)
    config.freeze()
    print(f"local rank {config.LOCAL_RANK} / global rank {dist.get_rank()} successfully build train dataset")
    if config.AUG.SSL_LINEAR_AUG:
        dataset_val, _ = build_dataset(is_train=False, config=config)
        print(f"local rank {config.LOCAL_RANK} / global rank {dist.get_rank()} successfully build val dataset")

    num_tasks = dist.get_world_size()
    global_rank = dist.get_rank()
    if config.DATA.ZIP_MODE and config.DATA.CACHE_MODE == 'part':
        indices = np.arange(dist.get_rank(), len(dataset_train), dist.get_world_size())
        sampler_train = SubsetRandomSampler(indices)
    else:
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
    if config.AUG.SSL_LINEAR_AUG:
        indices = np.arange(dist.get_rank(), len(dataset_val), dist.get_world_size())
        sampler_val = SubsetRandomSampler(indices)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=config.DATA.BATCH_SIZE,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=config.DATA.PIN_MEMORY,
        drop_last=True,
    )

    if config.AUG.SSL_LINEAR_AUG:
        data_loader_val = torch.utils.data.DataLoader(
            dataset_val, sampler=sampler_val,
            batch_size=config.DATA.BATCH_SIZE,
            shuffle=False,
            num_workers=config.DATA.NUM_WORKERS,
            pin_memory=config.DATA.PIN_MEMORY,
            drop_last=False
        )

    # setup mixup / cutmix
    mixup_fn = None
    mixup_active = config.AUG.MIXUP > 0 or config.AUG.CUTMIX > 0. or config.AUG.CUTMIX_MINMAX is not None
    if mixup_active:
        mixup_fn = Mixup(
            mixup_alpha=config.AUG.MIXUP, cutmix_alpha=config.AUG.CUTMIX, cutmix_minmax=config.AUG.CUTMIX_MINMAX,
            prob=config.AUG.MIXUP_PROB, switch_prob=config.AUG.MIXUP_SWITCH_PROB, mode=config.AUG.MIXUP_MODE,
            label_smoothing=config.MODEL.LABEL_SMOOTHING, num_classes=config.MODEL.NUM_CLASSES)
    if config.AUG.SSL_LINEAR_AUG:
        return dataset_train, dataset_val, data_loader_train, data_loader_val, mixup_fn
    else:
        return dataset_train, data_loader_train, mixup_fn


def build_dataset(is_train, config):
    transform = build_transform(is_train, config)

    if config.DATA.DATASET == 'imagenet':
        prefix = 'train' if is_train else 'val'
        if config.DATA.ZIP_MODE:
            ann_file = prefix + "_map.txt"
            prefix = prefix + ".zip@/"
            dataset = CachedImageFolder(config.DATA.DATA_PATH, ann_file, prefix, transform,
                                        cache_mode=config.DATA.CACHE_MODE if is_train else 'part')
        else:
            # ToDo: test custom_image_folder
            root = os.path.join(config.DATA.DATA_PATH, prefix)
            dataset = CustomImageFolder(root, transform=transform)
        nb_classes = 1000

    elif config.DATA.DATASET == 'wsi':
        prefix = 'train' if is_train else 'val'
        root = os.path.join(config.DATA.DATA_PATH, prefix)
        dataset = CustomImageFolder(root, config, transform=transform)
        nb_classes = 1

    else:
        raise NotImplementedError("We only support ImageNet or WSI Now.")

    return dataset, nb_classes
    
def build_transform(is_train, config):
    if config.AUG.SSL_AUG:
        if config.AUG.SSL_AUG_TYPE == 'byol':
            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            # if config.AUG.TRANSFORMATION == 'strap':
            #     #print("Transformation STRAP")
            #     T = stylize.StyleTransfer(style_dir=config.AUG.STRAP_STYLE_DIR, decoder_path=config.AUG.STRAP_DECODER_PATH, vgg_path=config.AUG.STRAP_VGG_PATH)

            if config.AUG.TRANSFORMATION == 'stain_aug':
            # elif config.AUG.TRANSFORMATION == 'stain_aug':
                T = stain_augment()
            elif config.AUG.TRANSFORMATION == 'stain_norm':
                T = stain_norm(config)

            if config.AUG.TRANSFORMATION in ['stain_aug', 'stain_norm']:  
            # if config.AUG.TRANSFORMATION is not None: 
                transform_1 = transforms.Compose([
                    T,
                    transforms.ToPILImage(),
                    transforms.RandomResizedCrop(config.DATA.IMG_SIZE, scale=(config.AUG.SSL_AUG_CROP, 1.)),
                    transforms.RandomHorizontalFlip(),
                    # transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
                    # transforms.RandomGrayscale(p=0.2),
                    transforms.RandomApply([GaussianBlur()], p=1.0),
                    transforms.ToTensor(),
                    normalize,
                ])
                transform_2 = transforms.Compose([
                    T,
	            transforms.ToPILImage(),
                    transforms.RandomResizedCrop(config.DATA.IMG_SIZE, scale=(config.AUG.SSL_AUG_CROP, 1.)),
                    transforms.RandomHorizontalFlip(),
                    # transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
                    # transforms.RandomGrayscale(p=0.2),
                    transforms.RandomApply([GaussianBlur()], p=0.1),
                    # transforms.RandomApply([ImageOps.solarize], p=0.2),
                    transforms.ToTensor(),
                    normalize,
                ])
            elif config.AUG.TRANSFORMATION == 'strap':
                # transform_1 = transforms.Compose([
                #     transforms.RandomResizedCrop(config.DATA.IMG_SIZE, scale=(config.AUG.SSL_AUG_CROP, 1.)),
                #     transforms.RandomHorizontalFlip(),
                #     # transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
                #     # transforms.RandomGrayscale(p=0.2),
                #     transforms.RandomApply([GaussianBlur()], p=1.0),
                #     transforms.ToTensor(),
                #     normalize,
                # ])
                transform_1 = transforms.Compose([
                    transforms.RandomResizedCrop(config.DATA.IMG_SIZE, scale=(config.AUG.SSL_AUG_CROP, 1.)),
                    transforms.RandomHorizontalFlip(),
                    # transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
                    #transforms.RandomGrayscale(p=0.2),
                    transforms.RandomApply([GaussianBlur()], p=1.0),
                    #transforms.RandomApply([ImageOps.solarize], p=0.2),
                    transforms.ToTensor(),
                    normalize,
                ])
                transform_2 = transforms.Compose([
                    transforms.RandomResizedCrop(config.DATA.IMG_SIZE, scale=(config.AUG.SSL_AUG_CROP, 1.)),
                    transforms.RandomHorizontalFlip(),
                    # transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
                    #transforms.RandomGrayscale(p=0.2),
                    transforms.RandomApply([GaussianBlur()], p=0.1),
                    # transforms.RandomApply([ImageOps.solarize], p=0.2),
                    transforms.ToTensor(),
                    normalize,
                ])
            elif config.AUG.TRANSFORMATION == 'low_pass':
                transform_1 = transforms.Compose([
                    low_pass(),
                    transforms.ToPILImage(),
                    transforms.RandomResizedCrop(config.DATA.IMG_SIZE, scale=(config.AUG.SSL_AUG_CROP, 1.)),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
                    transforms.RandomGrayscale(p=0.2),
                    transforms.RandomApply([GaussianBlur()], p=1.0),
                    transforms.ToTensor(),
                    normalize,
                ])
                transform_2 = transforms.Compose([
                    low_pass(),
                    transforms.ToPILImage(),
                    transforms.RandomResizedCrop(config.DATA.IMG_SIZE, scale=(config.AUG.SSL_AUG_CROP, 1.)),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
                    transforms.RandomGrayscale(p=0.2),
                    transforms.RandomApply([GaussianBlur()], p=0.1),
                    transforms.RandomApply([ImageOps.solarize], p=0.2),
                    transforms.ToTensor(),
                    normalize,
                ])
            else:
                transform_1 = transforms.Compose([
                    transforms.RandomResizedCrop(config.DATA.IMG_SIZE, scale=(config.AUG.SSL_AUG_CROP, 1.)),
                    transforms.RandomHorizontalFlip(),
                    # comment out two lines below
                    #transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
                    #transforms.RandomGrayscale(p=0.2),
                    transforms.RandomApply([GaussianBlur()], p=1.0),
                    transforms.ToTensor(),
                    normalize,
                ])
                transform_2 = transforms.Compose([
                    transforms.RandomResizedCrop(config.DATA.IMG_SIZE, scale=(config.AUG.SSL_AUG_CROP, 1.)),
                    transforms.RandomHorizontalFlip(),
                    # comment out two lines below
                    #transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
                    #transforms.RandomGrayscale(p=0.2),
                    transforms.RandomApply([GaussianBlur()], p=0.1),
                    # comment out one line below
                    #transforms.RandomApply([ImageOps.solarize], p=0.2),
                    transforms.ToTensor(),
                    normalize,
                ])

             
            transform = (transform_1, transform_2)
            return transform
        else:
            raise NotImplementedError
    
    if config.AUG.SSL_LINEAR_AUG:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
        if is_train:
            transform = transforms.Compose([
                transforms.RandomResizedCrop(config.DATA.IMG_SIZE),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize(config.DATA.IMG_SIZE + 32),
                transforms.CenterCrop(config.DATA.IMG_SIZE),
                transforms.ToTensor(),
                normalize,
            ])
        return transform

    resize_im = config.DATA.IMG_SIZE > 32
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=config.DATA.IMG_SIZE,
            is_training=True,
            color_jitter=config.AUG.COLOR_JITTER if config.AUG.COLOR_JITTER > 0 else None,
            auto_augment=config.AUG.AUTO_AUGMENT if config.AUG.AUTO_AUGMENT != 'none' else None,
            re_prob=config.AUG.REPROB,
            re_mode=config.AUG.REMODE,
            re_count=config.AUG.RECOUNT,
            interpolation=config.DATA.INTERPOLATION,
        )
        if not resize_im:
            # replace RandomResizedCropAndInterpolation with
            # RandomCrop
            transform.transforms[0] = transforms.RandomCrop(config.DATA.IMG_SIZE, padding=4)
        return transform

    t = []
    if resize_im:
        if config.TEST.CROP:
            size = int((256 / 224) * config.DATA.IMG_SIZE)
            t.append(
                transforms.Resize(size, interpolation=_pil_interp(config.DATA.INTERPOLATION)),
                # to maintain same ratio w.r.t. 224 images
            )
            t.append(transforms.CenterCrop(config.DATA.IMG_SIZE))
        else:
            t.append(
                transforms.Resize((config.DATA.IMG_SIZE, config.DATA.IMG_SIZE),
                                  interpolation=_pil_interp(config.DATA.INTERPOLATION))
            )

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    return transforms.Compose(t)

class GaussianBlur(object):
    """Gaussian Blur version 2"""

    def __call__(self, x):
        sigma = np.random.uniform(0.1, 2.0)
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x

class stain_augment(object):
    def __call__(self, sample):
        sample = np.array(sample)
        rgbaug = rgb_perturb_stain_concentration(sample, sigma1=1., sigma2=1.)
        return rgbaug
    
class stain_norm(object):
    def __init__(self, config):
        self.config = config
        
    def get_stain_normalizer(self, method='macenko'):
        target = staintools.read_image(self.config.AUG.STAIN_NORM_REF_PATH)
        target = staintools.LuminosityStandardizer.standardize(target)
        normalizer = staintools.StainNormalizer(method=method)
        normalizer.fit(target)
        return normalizer

    def apply_stain_norm(self, tile, normalizer):
        to_transform = np.array(tile).astype('uint8')
        to_transform = staintools.LuminosityStandardizer.standardize(to_transform)
        try:
            transformed = normalizer.transform(to_transform)
        except Exception:  # TissueMaskException
            return tile
        return transformed

    def __call__(self, sample):
        sample = np.array(sample)
        normalizer = self.get_stain_normalizer()
        rgb_normalized = self.apply_stain_norm(sample, normalizer)
        return rgb_normalized

class low_pass(object):
    def fft(self, img):
        return np.fft.fft2(img)

    def fftshift(self, img):
        return np.fft.fftshift(self.fft(img))

    def ifft(self, img):
        return np.fft.ifft2(img)

    def ifftshift(self, img):
        return self.ifft(np.fft.ifftshift(img))

    def distance(self, i, j, imageSize, r):
        dis = np.sqrt((i - imageSize/2) ** 2 + (j - imageSize/2) ** 2)
        if dis < r:
            return 1.0
        else:
            return 0
        
    def mask_radial(self, img):
        rows, cols = img.shape
        mask = np.zeros((rows, cols))
        r = rows//4
        for i in range(rows):
            for j in range(cols):
                mask[i, j] = self.distance(i, j, imageSize=rows, r=r)
        return mask

    def normalize(self, arr):
        new_arr = ((arr - arr.min()) * (1/(arr.max() - arr.min()) * 255)).astype('uint8')
        return new_arr

    def __call__(self, sample):
        sample = np.array(sample)
        tmp_low = np.zeros([sample.shape[0], sample.shape[1], 3])
        mask = self.mask_radial(np.zeros([sample.shape[0], sample.shape[1]]))
        for j in range(3):
            fd = self.fftshift(sample[:, :, j])
            fd_low = fd * mask
            img_low = self.ifftshift(fd_low)
            tmp_low[:,:,j] = np.real(img_low)
        return self.normalize(tmp_low)
