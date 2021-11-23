#@author: neel04
# General
from tqdm import tqdm
import os
import cv2
import numpy as np
import random

# Torch Imports
from torch.utils.data import DataLoader, BufferedShuffleDataset
from matplotlib import pyplot as plt
from torchvision.utils import save_image

# File Imports
from hvec import execute_shell, hevc_to_frames
from network import Net
from calib_dataset import CalibrationImageDataset

if not os.path.isdir('/content/calib-challenge-attempt/calib-challenge'):
    #Should be used on a fresh run
    execute_shell("git clone https://github.com/commaai/calib_challenge.git")
    execute_shell('git config --global user.email "neelgupta04@outlook.com"')
    execute_shell('git config --global user.name "neel04"')

if not os.path.isdir('/content/calib-challenge-attempt/data_3'):
    # Constructing the Image dataset from HEVC files    
    for i in tqdm(range(0,5)):
        hevc_to_frames(i, f'./data_{i}')

print(f'\nData Processing Complete! HVEC --> JPG\n')

BATCH_SIZE = 2

#PyTorch Dataset creation
train_ds = CalibrationImageDataset('/content/calib-challenge-attempt/', files=[0,1,2,3])
train_dataloader = DataLoader(train_ds, batch_size=BATCH_SIZE) #Making A dataloader from the fist 4 hvecs

val_ds = CalibrationImageDataset('/content/calib-challenge-attempt/', files=[4])
val_dataloader = DataLoader(val_ds, batch_size=BATCH_SIZE)              #Making A dataloader from the fist 4 hvecs

#print(f'\nTrain Samples: {len(train_dataloader)}\nVal Samples: {len(val_dataloader)}\nBatch Size: {BATCH_SIZE}')

img_1, tgt_1 = next(iter(train_dataloader))
save_image(img_1[0], f'./train_sample.jpg')

my_nn = Net()
print('Model Architecture:\n',my_nn, end='\n')

#======CLEANUP===========
#Before Committing
# rm -rf ./data*