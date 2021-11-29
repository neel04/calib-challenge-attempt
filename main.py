#@author: neel04
# General
from tqdm import tqdm
import os
from torchinfo import summary
from pytorch_lightning.loggers import WandbLogger

# Torch Imports
from torch.utils.data import DataLoader
from pytorch_lightning.loggers import TensorBoardLogger
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

BATCH_SIZE = 64

train_ds = CalibrationImageDataset('/content/calib-challenge-attempt/', files=[0,1,2,3])
train_dataloader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True) #Making A dataloader from the fist 4 hvecs

print(f'Train Samples: {len(train_dataloader)}\nBatch Size: {BATCH_SIZE}\n')

#img_1, tgt_1 = next(iter(train_dataloader))
#save_image(img_1[0], f'./train_sample.jpg')
#print(img_1.shape)  # torch.Size([2, 337, 1164])

#Create the model and train it
model = CalibNet(input_size=(256,512), output_size=1, hidden_size=1024, batch_size=BATCH_SIZE, lr=0.01) #Adjust Hidden Size
summary(model, input_size=(BATCH_SIZE, 1, 256, 512))
print(model)

# Initializing Trainer
pl.seed_everything(420)
trainer = pl.Trainer(max_epochs=100, gpus=1, accelerator='gpu', logger=[TensorBoardLogger('./tb_logs', name='CalibNet'), WandbLogger(project="CalibNet")],
                    precision=16)

trainer.fit(model)

#======CLEANUP===========
#Before Committing
# rm -rf ./data*
