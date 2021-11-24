#@author: neel04
# General
from tqdm import tqdm
import os

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
train_dataloader = DataLoader(train_ds, batch_size=self.BATCH_SIZE) #Making A dataloader from the fist 4 hvecs

#print(f'\nTrain Samples: {len(train_dataloader)}\nVal Samples: {len(val_dataloader)}\nBatch Size: {BATCH_SIZE}')

img_1, tgt_1 = next(iter(train_dataloader))
save_image(img_1[0], f'./train_sample.jpg')

#Create the model and train it
model = CalibNet(input_size=(376,1196), output_size=(2,), hidden_size=50, batch_size=BATCH_SIZE)
trainer = pl.Trainer(max_epochs=10)
trainer.fit(model)

#======CLEANUP===========
#Before Committing
# rm -rf ./data*