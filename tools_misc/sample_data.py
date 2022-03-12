import argparse
import os
import random
import shutil
from pathlib import Path

parser = argparse.ArgumentParser(description='Check data. ')
parser.add_argument('--input_dir', type=str, help='File path')
parser.add_argument('--output_dir', type=str, help='File path')
parser.add_argument('--number_samples', type=int, help='number of samples')
parser.add_argument('--seed', type=int, help='seed')

args = parser.parse_args()

path = Path(args.output_dir)
path.mkdir(parents=True, exist_ok=True)

random.seed(args.seed)
image_list = os.listdir(args.input_dir)
random.shuffle(image_list)
sample_set = image_list[:args.number_samples]
for sample in sample_set:
    src = os.path.join(args.input_dir, sample)
    dst = os.path.join(args.output_dir, sample)
    #print(src)
    #print(dst)
    shutil.copy(src, args.output_dir)
