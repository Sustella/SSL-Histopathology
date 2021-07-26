import sys, argparse, os, glob
import tifffile
import math
import numpy as np
from PIL import Image
import openslide
#import staintools

from preprocess import apply_image_filters, tissue_percent
from itertools import product


def get_downsampled_image(slide, target_level, scale_factor=32):
    large_w, large_h=slide.level_dimensions[target_level]
    #large_w, large_h=slide.dimensions
    #print('target slide.dimensions width {} height {}'.format(large_w, large_h))
    downsample_level = slide.get_best_level_for_downsample(scale_factor)
    if downsample_level == target_level:
        s = 1
    else:
        s = scale_factor/(target_level+1)
    new_w = math.floor(large_w/s)
    new_h = math.floor(large_h/s)    
    downsample_level = slide.get_best_level_for_downsample(scale_factor)
    #print("downsample_level: {}".format(downsample_level))
    whole_slide_image = slide.read_region((0, 0), downsample_level, slide.level_dimensions[downsample_level])
    #whole_slide_image = slide.read_region((0, 0), 1, slide.level_dimensions[1])
    whole_slide_image = whole_slide_image.convert("RGB")
    img = whole_slide_image.resize((new_w, new_h), Image.BILINEAR)
    return img, new_w, new_h, s

def get_start_end_coordinates(x, tile_size):
    start = int(x * tile_size)
    end = int((x+1) * tile_size)
    return start, end

def get_stain_normalizer(path='/scratch/users/stellasu/stain_normalization/stain_normalize_target.png', method='macenko'):
    target = staintools.read_image(path)
    target = staintools.LuminosityStandardizer.standardize(target)
    normalizer = staintools.StainNormalizer(method=method)
    normalizer.fit(target)
    return normalizer

def apply_stain_norm(tile, normalizer):
    to_transform = np.array(tile).astype('uint8')
    to_transform = staintools.LuminosityStandardizer.standardize(to_transform)
    transformed = normalizer.transform(to_transform)
    transformed = Image.fromarray(transformed)
    return transformed



def main():
    parser = argparse.ArgumentParser(description='Tiling Panda Grand Challenge Dataset')
    parser.add_argument('--dataset_path', default='/scratch/users/stellasu/prostate_panda/train_images/', type=str, help='Input dataset path')
    parser.add_argument('--output_path', default='/scratch/users/stellasu/prostate_panda/train_images_tiles/', type=str, help='Output dataset path')
    parser.add_argument('--tile_size', default=256, type=int, help='Tile Size')
    parser.add_argument('--level', default=0, type=int, help='Level of image, magnification')
    args = parser.parse_args()
   
    # For testing
    slide_rootpath='/scratch/users/stellasu/prostate_panda/train_images/'
    out_rootpath='/scratch/users/stellasu/prostate_panda/train_images_tiles_test/'
    # For tcga_msi_wsi
#     slide_rootpath='/scratch/users/stellasu/tcga_msi_wsi/'
#     out_rootpath='/scratch/users/stellasu/tcga_msi_wsi_tiles/'
#     os.path.join(data_dir, f'{case_id}.tiff')
    slide_list = [os.path.split(i)[1][:-5] for i in glob.glob(args.dataset_path+'*tiff')]
#     print(slide_list)

    tile_size = args.tile_size
    
    level = args.level
    scale_factor = 32 
    
    #print('scale_factor: {}'.format(scale_factor))
#     normalizer = get_stain_normalizer()
    levelpath = args.output_path+str(level) + '/'
    if not os.path.exists(levelpath):
        os.makedirs(levelpath)
    for i in range(len(slide_list)):
        savepath = levelpath + slide_list[i] + '/'
        slide_path=args.dataset_path + slide_list[i] + '.tiff'
        slide = openslide.open_slide(slide_path)
        if not os.path.exists(savepath):
            os.makedirs(savepath)
        
        img, new_w, new_h, s = get_downsampled_image(slide, level, scale_factor=scale_factor)
        #print('new_w {} new_h {} s {}'.format(new_w, new_h, s))
        tissue=apply_image_filters(np.array(img))

        small_tile_size = int(((tile_size/s)*2+1)//2)
        num_tiles_h = new_h//small_tile_size
        num_tiles_w = new_w//small_tile_size
        #print('num_tiles_h {}'.format(num_tiles_h))
        #print('num_tiles_w {}'.format(num_tiles_w))

        for h in range(num_tiles_h):
            for w in range(num_tiles_w):
                small_start_h, small_end_h = get_start_end_coordinates(h, small_tile_size)
                small_start_w, small_end_w = get_start_end_coordinates(w, small_tile_size)
                tile_region = tissue[small_start_h:small_end_h, small_start_w:small_end_w]
                if tissue_percent(tile_region)>=75:
                    try:
                        start_h, end_h = get_start_end_coordinates(h, tile_size)
                        start_w, end_w = get_start_end_coordinates(w, tile_size)
                        #print('start_w {} small_start_w {}'.format(start_w, small_start_w))
                        #print('start_h {} small_start_h {}'.format(start_h, small_start_h))
                        tile_path = savepath+slide_list[i]+'_'+str(tile_size)+'_x'+str(start_w)+'_'+str(w)+'_'+str(num_tiles_w)+'_y'+str(start_h)+'_'+str(h)+'_'+str(num_tiles_h)+'.png'
                        if os.path.exists(tile_path):
                            print('%s Already Tiled' % (tile_path))
                        else:
                            tile = slide.read_region((start_w*4**level, start_h*4**level), level, (tile_size, tile_size))
                            tile = tile.convert("RGB")
                        #    im = Image.fromarray(tile_region)
                        #    im.save(tile_path)
                            tile.save(tile_path)
                        #print(tile_region.shape)
                    except:
                        print('error for %s' % (tile_path)) 
        print('Done for %s' % slide_list[i])

if __name__ == '__main__':
    main()
