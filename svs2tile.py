import sys
import os
import glob
import argparse
import math
import random
import numpy as np
from PIL import Image
import openslide

from preprocess import apply_image_filters, tissue_percent, isWhitePatch, isBlackPatch


def get_downsampled_image(slide, scale_factor=32):
    large_w, large_h=slide.dimensions
    new_w = math.floor(large_w/scale_factor)
    new_h = math.floor(large_h/scale_factor)    
    level = slide.get_best_level_for_downsample(scale_factor)
    whole_slide_image = slide.read_region((0, 0), level, slide.level_dimensions[level])
    whole_slide_image = whole_slide_image.convert("RGB")
    img = whole_slide_image.resize((new_w, new_h), Image.BILINEAR)
    return img, new_w, new_h

def get_start_end_coordinates(x, tile_size):
    start = int(x * tile_size)
    end = int((x+1) * tile_size)
    return start, end

def main():
    parser = argparse.ArgumentParser(description='Tiling WSI Datasets')
    parser.add_argument('--data_root_path', default='/scratch/users/rikiya/ssl_datasets/', type=str, help='Input root path')
    parser.add_argument('--output_root_path', default='/scratch/users/rikiya/ssl_tiles/', type=str, help='Output root path')
    parser.add_argument('--dataset_path', default='ctpac/ccrcc/', type=str, help='Dataset path')
    parser.add_argument('--tile_size', default=224, type=int, help='Tile size')
    parser.add_argument('--level', default=0, type=int, help='Magnification level')
    # parser.add_argument('--start', default=0, type=int, help='Start index')
    # parser.add_argument('--stop', default=10, type=int, help='Stop index')
    args = parser.parse_args()

    """
    'ctpac/ccrcc',
    'ctpac/gbm',
    'ctpac/coad',
    'ctpac/lscc',
    'ctpac/brca',
    'ctpac/ov',
    'ctpac/sar',
    'ctpac/cm',
    'ctpac/luad',
    'ctpac/pda',
    'ctpac/ucec',
    'ctpac/hnscc',
    'camelyon16/training/normal',
    'camelyon16/training/tumor',
    'camelyon16/testing/images',
    'camelyon17/training/center_0',
    'camelyon17/training/center_1',
    'camelyon17/training/center_2',
    'camelyon17/training/center_3',
    'camelyon17/training/center_4',
    'camelyon17/testing/patients',
    'slnbreast',
    'panda/train_images',
    'bach/ICIAR2018_BACH_Challenge/WSI',
    'bach/ICIAR2018_BACH_Challenge_TestDataset/WSI',
    'tupac16/training_image_data',
    'tupac16/testing_image_data'
    """
    
    slide_rootpath = args.data_root_path + args.dataset_path
    out_rootpath = args.output_root_path + args.dataset_path
    dataset_name = '_'.join(args.dataset_path[:-1].split('/'))
    
    extensions = ['svs', 'tif', 'tiff']
    slide_list = []
    for ext in extensions:
        tmp = glob.glob(f"{slide_rootpath}*.{ext}")
        if len(tmp) > 0:
            for j in tmp:
                slide_list.append(j)
    slide_list = sorted(slide_list)
    # to get filename: os.path.splitext(os.path.basename(slide_list[i]))[0]
    
    tile_size = args.tile_size
    level = args.level
    
    # start = args.start
    # stop = args.stop
    # if int(stop) > len(slide_list):
    #     stop = str(len(slide_list))

    scale_factor = 32

    # for i in range(int(start), int(stop)):
    for i in range(len(slide_list)):
        counter = 0
        
        savepath = f"{out_rootpath}{os.path.splitext(os.path.basename(slide_list[i]))[0]}"
        if not os.path.exists(savepath):
            os.makedirs(savepath) 

        slide = openslide.open_slide(slide_list[i])
        prop = slide.properties
        if 'openslide.mpp-x' in prop:
            mpp = round(float(prop['openslide.mpp-x']),3)
        elif 'tiff.XResolution' in prop:
            mpp = round(1 / (float(prop['tiff.XResolution']) / 10000),3)
        else:
            mpp = 0
        downsample_factor = slide.level_downsamples

        try:
            img, new_w, new_h = get_downsampled_image(slide, scale_factor=scale_factor)
        except:
            continue

        tissue = apply_image_filters(np.array(img))

        small_tile_size = int(((tile_size/scale_factor)*2+1)//2)
        num_tiles_h = new_h//small_tile_size
        num_tiles_w = new_w//small_tile_size

        coordinates = []
        for h in range(num_tiles_h):
            for w in range(num_tiles_w):
                small_start_h, small_end_h = get_start_end_coordinates(h, small_tile_size)
                small_start_w, small_end_w = get_start_end_coordinates(w, small_tile_size)
                tile_region = tissue[small_start_h:small_end_h, small_start_w:small_end_w]
                if tissue_percent(tile_region) >= 90:
                    coordinates.append((h, w))
        
        random.seed(42)
        if 'panda' in dataset_name:
            if len(coordinates) < 20:
                sample_coordinates = coordinates
            else:
                sample_coordinates = random.sample(coordinates, k=20)
        else:
            if len(coordinates) < 200:
                sample_coordinates = coordinates
            else:
                sample_coordinates = random.sample(coordinates, k=200)
        for h, w in sample_coordinates:
            if ('panda' in dataset_name) and (counter == 10):
                break
            if ('panda' not in dataset_name) and (counter == 100):
                break
            # small_start_h, small_end_h = get_start_end_coordinates(h, small_tile_size)
            # small_start_w, small_end_w = get_start_end_coordinates(w, small_tile_size)
            # tile_region = tissue[small_start_h:small_end_h, small_start_w:small_end_w]
            # if tissue_percent(tile_region) >= 90:
            # try:
            start_h, end_h = get_start_end_coordinates(h, tile_size)
            start_w, end_w = get_start_end_coordinates(w, tile_size)
            tile_path = f"{savepath}/{dataset_name}_{os.path.splitext(os.path.basename(slide_list[i]))[0]}_{tile_size}_x{start_w}_{w}_{num_tiles_w}_y{start_h}_{h}_{num_tiles_h}_mpp_{mpp}.png"
            if os.path.exists(tile_path):
                print('%s Alraedy Tiled' % (tile_path))
            else:
                tile = slide.read_region((start_w*int(round(downsample_factor[level],0)), start_h*int(round(downsample_factor[level],0))), level, (tile_size, tile_size))
                tile = tile.convert("RGB")
                if not isWhitePatch(np.array(tile)) and not isBlackPatch(np.array(tile)):
                    tile.save(tile_path)
                    counter += 1
            # except:
            #     print(f'error for {tile_path}')
                        
        print(f'Done for {os.path.splitext(os.path.basename(slide_list[i]))[0]}: total {counter} tiles saved')
        
if __name__ == "__main__":
    main()