import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

parser = argparse.ArgumentParser(description='Plot data. ')
parser.add_argument('--filepath', type=str, help='File path')

args = parser.parse_args()
print(args.filepath)

df = pd.read_csv(args.filepath, header=None)
print(df.head())
y = df[0].values.tolist()
#print(len(y))
min_y = min(y)
index_min_y = y.index(min_y)
print('min_y: {} index_min_y: {}'.format(min_y, index_min_y))
x = np.arange(0, len(y), 1)
plt.plot(x, y, marker='.', label='Loss during training')
plt.title('Training loss during SSL training')
plt.xlabel('Epoch')
plt.ylabel('Training loss')
#plt.show()
plt.savefig(args.filepath[:-4] + '.png')

