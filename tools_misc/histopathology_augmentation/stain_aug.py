import os
import numpy as np
from PIL import Image
from histomicstk.preprocessing.augmentation.color_augmentation import rgb_perturb_stain_concentration
import argparse
from numpy import asarray

parser = argparse.ArgumentParser(description='Run stain aug.')
parser.add_argument('--batch_num', type=int, help='batch number')

args = parser.parse_args()
print(args.batch_num)

path = '/scratch/groups/rubin/stellasu/ssl_pretrain_tiny/train/0/'
#path = '/home/users/stellasu/histopathology_augmentation/camelyon17_v1.0_for_figures/'
dataset = os.listdir(path)
dataset.sort()
print(len(dataset))
batch_size = 10000

start_idx = (args.batch_num-1)*batch_size
end_idx = start_idx + batch_size
end_idx = min(end_idx, len(dataset))
print(start_idx)
print(end_idx)
print(end_idx-start_idx)
exclusion_list = ['ADI-GSGCQMCR.tif', 'ADI-MDWIYNNP.tif', 'ADI-PVELLPHE.tif', 'ADI-RIHMVCIE.tif', 'MUC-LQFIVSTF.tif', 'camelyon17_testing_patients_patient_149_node_0_224_x72800_325_431_y8512_38_880_mpp_0.243.png', 'camelyon17_testing_patients_patient_179_node_1_224_x79968_357_423_y20608_92_940_mpp_0.243.png', 'camelyon17_training_center_0_patient_004_node_1_224_x94528_422_431_y113120_505_880_mpp_0.243.png', 'camelyon17_training_center_3_patient_068_node_3_224_x32256_144_431_y3360_15_880_mpp_0.243.png', 'ctpac_hnscc_C3N-04273-25_224_x3584_16_391_y11424_51_87_mpp_0.494.png', 'ctpac_pda_C3N-03000-22_224_x60032_268_382_y10752_48_50_mpp_0.494.png', 'patch_patient_009_node_1_x_12096_y_34656_0_1'] 

for i in range(start_idx, end_idx):
#for i in range(79261, end_idx):
    f = dataset[i]
    print('i: {} f: {}'.format(i, f))
    #if f != 'ADI-GSGCQMCR.tif':
    #    continue
    if f in exclusion_list:
        continue
    # load the image
    image = Image.open(path+f)
    # convert image to numpy array
    data = asarray(image)
    #print(type(data))
    # summarize shape
    #print(data.shape)
    rgbaug = rgb_perturb_stain_concentration(data, sigma1=1., sigma2=1.)
