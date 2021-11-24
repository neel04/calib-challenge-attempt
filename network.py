# PyTorch modules and general stuff
from calib_dataset import CalibrationImageDataset

import torch
from torch.nn.modules.loss import MSELoss
import pytorch_lightning as pl
from torch import nn

class CalibNet(pl.LightningModule):
    def __init__(self, input_size, output_size, hidden_size, batch_size):
        super(CalibNet, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.BATCH_SIZE = batch_size
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(256*4*4, self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size, self.output_size)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.conv4(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = x.view(-1, 256*4*4)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x
    
    def training_step(self, batch, batch_idx):
        x, (y_1, y_2) = batch
        y_hat_1, y_hat_2 = self.forward(x)
        loss = MSELoss(y_hat_1, y_1) + MSELoss(y_hat_2, y_2)
        tensorboard_logs = {'train_loss': loss}
        self.log(f'train loss: {loss}')
        return {'loss': loss, 'log': tensorboard_logs}

    def MAPELoss(self, output, target):
      return torch.mean(torch.abs((target - output) / target))  
    
    def validation_step(self, batch, batch_idx):
        x, (y_1, y_2) = batch
        y_hat_1, y_hat_2 = self.forward(x)
        mape_loss = self.MAPELoss(y_hat_1, y_1) + self.MAPELoss(y_hat_2, y_2)

        tensorboard_logs = {'MAPE:':mape_loss}
        return {'val_loss': mape_loss, 'log':tensorboard_logs}
    
    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        tensorboard_logs = {'val_loss': avg_loss}
        return {'val_loss': avg_loss, 'log': tensorboard_logs}
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        return optimizer
    
    def train_dataloader(self):
        #Making A dataloader from the fist 4 hvecs
        train_ds = CalibrationImageDataset('/content/calib-challenge-attempt/', files=[0,1,2,3])
        train_dataloader = (train_ds, batch_size=self.BATCH_SIZE) #Making A dataloader from the fist 4 hvecs
        return train_dataloader
    
    def val_dataloader(self):
      #Making A dataloader from the last file
      val_ds = CalibrationImageDataset('/content/calib-challenge-attempt/', files=[4])
      val_dataloader = torch.utils.data.DataLoader(val_ds, batch_size=self.BATCH_SIZE)