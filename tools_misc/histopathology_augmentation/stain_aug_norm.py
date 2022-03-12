import os
import numpy as np
from PIL import Image
from histomicstk.preprocessing.augmentation.color_augmentation import rgb_perturb_stain_concentration
import staintools
import argparse
from numpy import asarray

parser = argparse.ArgumentParser(description='Run stain aug.')
parser.add_argument('--filename', type=str, help='filename')
parser.add_argument('--transform_type', type=str, help='transform type: stain_aug or stain_norm')

args = parser.parse_args()
print(args.filename)
print(args.transform_type)

#path = '/scratch/groups/rubin/stellasu/ssl_pretrain_tiny/train/0/'
path = '/scratch/groups/rubin/stellasu/wilds/camelyon17_v1.0/val/1/'

# load the image
image = Image.open(path+args.filename)
# convert image to numpy array
data = asarray(image)
data = data[:,:,:3]
#print(type(data))
# summarize shape
#print(data.shape)
if args.transform_type == 'stain_aug':
    rgbaug = rgb_perturb_stain_concentration(data, sigma1=1., sigma2=1.)
print(type(rgbaug))

