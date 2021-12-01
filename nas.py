from torch.utils import data
from network import CalibNet
from calib_dataset import CalibrationImageDataset
from hvec import execute_shell, hevc_to_frames

import tensorflow as tf
import torch
import os
from tqdm import tqdm
import nonechucks as nc

#============================================================================================================
# Constructing the files for the dataset
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
#============================================================================================================

train_ds = nc.SafeDataset(CalibrationImageDataset('/content/calib-challenge-attempt/', files=[0,1,4,3])) #Making A dataset from the fist 4 hvecs
val_ds = nc.SafeDataset(CalibrationImageDataset('/content/calib-challenge-attempt/', files=[2]))  #2 is slightly different, hence good test for generalization

def data_generator(batch_size, dataset):
    '''
    Dataset Generator for easily constructing tf.Dataset;
    adapter for a torch Dataset.
    
    returns (batch_sizes, 256, 512) + (batch_size, 2)
    '''
    dloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    for img, tgt in iter(dloader):
        tgt = [i.numpy() for i in tgt]
        img = img.numpy().reshape(batch_size, 256, 512)
        yield img, tgt

# Creating the generator
data_gen = data_generator(dataset=train_ds, batch_size=2)
    
for image, tgt in data_gen:
    print(image[0].shape)
    break
