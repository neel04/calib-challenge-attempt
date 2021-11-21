#@author: neel04
from tqdm import tqdm
import os
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

print(f'\nData Processing Complete! HVEC --> JPG\n')

#PyTorch Dataset creation
test_ds = CalibrationImageDataset('/content/calib-challenge-attempt/')
train_dataloader = DataLoader(test_ds)
img, tgt = next(iter(train_dataloader))

save_image(img.float(), f'./sanity_check.jpg')

#======CLEANUP===========
#Before Committing
# rm -rf ./data*