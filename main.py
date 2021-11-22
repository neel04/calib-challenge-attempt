#@author: neel04
from tqdm import tqdm
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from calib_dataset import CalibrationImageDataset
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from hvec import execute_shell, hevc_to_frames

if not os.path.isdir('/content/calib-challenge-attempt/calib-challenge'):
    #Should be used on a fresh run
    execute_shell("git clone https://github.com/commaai/calib_challenge.git")
    execute_shell('git config --global user.email "neelgupta04@outlook.com"')
    execute_shell('git config --global user.name "neel04"')

if not os.path.isdir('/content/calib-challenge-attempt/data_3'):
    # Constructing the Image dataset from HEVC files    
    for i in tqdm(range(0,5)):
        hevc_to_frames(i, f'./data_{i}')

print(f'\nData Processing Complete! HVEC --> JPG')

BATCH_SIZE = 2

#PyTorch Dataset creation
train_ds = CalibrationImageDataset('/content/calib-challenge-attempt/', files=[0,1,2,3])
train_dataloader = DataLoader(train_ds, batch_size=BATCH_SIZE)              #Making A dataloader from the fist 4 hvecs

val_ds = CalibrationImageDataset('/content/calib-challenge-attempt/', files=[4])
val_dataloader = DataLoader(val_ds, batch_size=BATCH_SIZE)              #Making A dataloader from the fist 4 hvecs

#print(f'\nTrain Samples: {len([tgt for img,tgt in train_dataloader])}\nVal Samples: {len([tgt for img, tgt in val_dataloader])}')

for img,tgt in train_dataloader:
    print(f'shape: {img[0].shape}')
    save_image(img[0].float(), './sample.jpg')
    break

for img, tgt in val_dataloader:
    cv2.imwrite('./test_val.jpg', img[0])
    break

#======CLEANUP===========
#Before Committing
# rm -rf ./data*