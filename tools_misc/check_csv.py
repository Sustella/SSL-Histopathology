import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from collections import Counter
import shutil
from pathlib import Path

parser = argparse.ArgumentParser(description='Check data. ')
parser.add_argument('--filepath', type=str, help='File path')
parser.add_argument('--input_dir', type=str, help='File path')
parser.add_argument('--output_dir', type=str, help='File path')
parser.add_argument('--center_no', type=int, help='Center No')
parser.add_argument('--label_no', type=int, help='label No')


args = parser.parse_args()
print(args.filepath)

df = pd.read_csv(args.filepath)
print(df.head())
print(df.columns)
data_split = df['data_split'].tolist()
print(Counter(data_split))
center_list = df['center'].tolist()
print(set(center_list))
df1 = df.loc[(df['center'] == args.center_no) & (df['tumor']==args.label_no) & (df['data_split']=='val')]
print(df1)
filename_df = df1['fnames']
filename_df.to_csv('center_'+str(args.center_no)+'_label_'+str(args.label_no)+'.csv', index=False)
file_list = filename_df.tolist()

path = Path(args.output_dir)
path.mkdir(parents=True, exist_ok=True)

for fn in file_list:
    p = fn.split('/')
    src = os.path.join(args.input_dir, p[-1])
    #print(src)
    shutil.copy(src, args.output_dir)
