
import os

dir_path = '/scratch/groups/rubin/stellasu/ssl_pretrain_tiny_stylized/train/0/'
files = os.listdir(dir_path)
print(len(files))

for f in files:
    sub_files = os.listdir(dir_path+f)
    #print(sub_files)
    if len(sub_files) != 32:
       print(dir_path+f)
       continue
    for f1 in sub_files:
        fn = dir_path+f+'/'+f1
        s = os.path.getsize(fn)
        if s == 0:
            print(fn)
       
