# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Zhenda Xie
# --------------------------------------------------------

import os, glob, random
from torchvision import datasets


class CustomImageFolder(datasets.ImageFolder):
    def __init__(self, root, config, *args, **kwargs):
        super().__init__(root, *args, **kwargs)
        self.config = config

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        #print('path: {}'.format(path))
        #print('target: {}'.format(target))

        if self.config.AUG.TRANSFORMATION == 'strap':
            img_name = os.path.basename(path)[:-4]
            #stylized_dir = f"/scratch/groups/rubin/stellasu/ssl_pretrain_tinier_stylized_rev/train/0/{img_name}/"
            stylized_dir = f"/scratch/groups/rubin/stellasu/ssl_pretrain_tiny_stylized/train/0/{img_name}/"
            stylized_imgs = glob.glob(stylized_dir+"*.png")
            if len(stylized_imgs) == 0:
                print('img_name: {}'.format(img_name))
            #print('number of stylized_imgs: {}'.format(len(stylized_imgs))) 
            #print('stylized_imgs: {}'.format(stylized_imgs))
    
            ##### BOTH STYLIZE #####
            #try:
                #print('both stylized')
            #    paths = random.sample(stylized_imgs, 2)
            #    images = [self.loader(p) for p in paths]
            #except:
            #    print(img_name)
            #    images = [self.loader(p) for p in stylized_imgs[:2]]
            ##########

            ##### HALF STYLIZE #####
            # print('half stylized')
            stylized = self.loader(random.choice(stylized_imgs))
            image = self.loader(path)
            images = [image, stylized]
            ##########

            ret = []
            if self.transform is not None:
                for i, t in zip(images, self.transform):
                    ret.append(t(i))
        else:
            image = self.loader(path)
            ret = []
            if self.transform is not None:
                for t in self.transform:
                    try:
                        ret.append(t(image))
                    except:
                        print('transform failure: {}'.format(path))
            else:
                ret.append(image)

        if self.target_transform is not None:
            target = self.target_transform(target)
        ret.append(target)

        return ret
