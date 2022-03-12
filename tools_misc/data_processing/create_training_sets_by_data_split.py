from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import shutil

def split(input_image_dir, file_path, dataset_output_dir, image_format):
  """Runs the split operation.

  Args:
  """
# patch_patient_004_node_4_x_3808_y_19648.png
# sample,patient,node,x_coord,y_coord,tumor,slide,center,split
  make_dir(dataset_output_dir)
  sample_label_dict = {}
  sample_folder_dict = {}
  sample_split_dict = {}
  label_list = []
  split_list = []
  with open(file_path) as f:
    next(f)
    examples_list = f.read().splitlines()
    for example in examples_list:
      sample_list = example.split(',')
      class_id = sample_list[5]
      split = sample_list[10]
      patient_id = "patient_{0:03}".format(int(sample_list[1]))
      #print(patient_id)
      sample_name = "patch_"+patient_id+"_node_"+sample_list[2]+"_x_"+sample_list[3]+"_y_"+sample_list[4]
      #print(sample_name)
      folder_name = patient_id+"_node_"+sample_list[2]
      #print(folder_name)
      sample_label_dict[sample_name] = class_id
      sample_folder_dict[sample_name] = folder_name
      sample_split_dict[sample_name] = split
      label_list.append(class_id)
      split_list.append(split)
      #print(class_id)
      #break
  labels = set(label_list)
  splits = set(split_list)
  for splt in splits:
    for label in labels:
      make_dir(os.path.join(dataset_output_dir,splt,label))
  for key, value in sample_label_dict.items():
    src = os.path.join(input_image_dir, "patches", sample_folder_dict[key], key + '.'+ image_format)
    #print(src)
    dst = os.path.join(dataset_output_dir, sample_split_dict[key], value, key + '.' + image_format)
    #print(dst)
    shutil.copyfile(src, dst)
  

def make_dir(dirName):
  if not os.path.exists(dirName):
    os.makedirs(dirName, exist_ok=True)


def main():
  parser = argparse.ArgumentParser(description='Splitting training images by labels')
  parser.add_argument('--input_image_dir', type=str, default='', 
          help='The directory where the input images are stored.') 
  parser.add_argument('--dataset_label_path', type=str,
          default='',
          help='The path of the dataset label file')
  parser.add_argument('--dataset_output_dir', type=str, default='',
          help='The root directory where the output are saved.')
  parser.add_argument('--image_format', type=str, default='',
          help='Image format.')
  args = parser.parse_args()
  
  split(args.input_image_dir, args.dataset_label_path, args.dataset_output_dir, args.image_format)

if __name__ == '__main__':
  main()
