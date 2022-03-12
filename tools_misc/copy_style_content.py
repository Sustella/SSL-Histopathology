import argparse
import os
import shutil
from pathlib import Path

parser = argparse.ArgumentParser(description='Check data. ')
parser.add_argument('--input_dir', type=str, help='File path')
parser.add_argument('--output_dir', type=str, help='File path')

args = parser.parse_args()
#file_number_list = [12, 14, 37, 50, 57, 59, 74, 83, 106, 122, 165, 187, 371, 689, 882, 1053, 1122, 3693, 554, 1805, 2059, 2156, 2185, 2439, 2859, 3252, 5676, 367, 373, 688, 743, 1357, 1429, 1837, 1891]
file_number_list = [55270, 55957, 56176, 56207, 56209, 57019, 57316]
path = Path(args.output_dir)
path.mkdir(parents=True, exist_ok=True)

for fn in file_number_list:
    src = os.path.join(args.input_dir, str(fn)+'.jpg')
    print(src)
    shutil.copy(src, args.output_dir)
