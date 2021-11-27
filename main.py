#@author: neel04
# General
from tqdm import tqdm
import os
from torchinfo import summary

# Torch Imports
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import pytorch_lightning as pl

# File Imports
from hvec import execute_shell, hevc_to_frames
from network import CalibNet
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

train_ds = CalibrationImageDataset('/content/calib-challenge-attempt/', files=[0,1,2,3])
train_dataloader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True) #Making A dataloader from the fist 4 hvecs

print(f'Train Samples: {len(train_dataloader)}\nBatch Size: {BATCH_SIZE}\n')

#img_1, tgt_1 = next(iter(train_dataloader))
#save_image(img_1[0], f'./train_sample.jpg')
#print(img_1.shape)  # torch.Size([2, 337, 1164])

#Create the model and train it
model = CalibNet(input_size=(256,512), output_size=1, hidden_size=512, batch_size=BATCH_SIZE, lr=0.001)
summary(model, input_size=(BATCH_SIZE, 1, 256, 512))

# Initializing Trainer
trainer = pl.Trainer(max_epochs=10, devices=1, accelerator='gpu', auto_lr_find=True)
trainer.fit(model)

#======CLEANUP===========
#Before Committing
# rm -rf ./data*