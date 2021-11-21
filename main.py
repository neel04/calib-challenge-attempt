#@author: neel04
from tqdm import tqdm
import os
import numpy as np
import matplotlib.pyplot as plt

from calib_dataset import CalibrationImageDataset
from torch.utils.data import DataLoader
from hvec import execute_shell, hevc_to_frames

execute_shell('rm -rf ./data*')

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

#PyTorch Dataset creation
test_ds = CalibrationImageDataset('/content/calib-challenge-attempt/')
train_dataloader = DataLoader(test_ds)

img, tgt = next(iter(train_dataloader))
print(np.float64(tgt))
plt.plot(img)

#======CLEANUP===========
#Before Committing
# rm -rf ./data*