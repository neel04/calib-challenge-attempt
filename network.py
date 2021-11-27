# PyTorch modules and general stuff
from calib_dataset import CalibrationImageDataset

import torch
from torch.nn.modules.loss import MSELoss
import pytorch_lightning as pl
from torch import nn
import nonechucks as nc

class CalibNet(pl.LightningModule):
    def __init__(self, input_size, output_size, hidden_size, batch_size, lr):
        super(CalibNet, self).__init__()
        self.lr = lr
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(4608, self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size, self.output_size)
        self.relu = torch.nn.ReLU()
        self.maxpool1 = torch.nn.MaxPool2d(stride=4, kernel_size=3)
        self.maxpool2 = torch.nn.MaxPool2d(stride=4, kernel_size=3)
        self.maxpool3 = torch.nn.MaxPool2d(stride=5, kernel_size=3)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.maxpool2(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.maxpool3(x)

        gate = x.view(-1, 4608)

        x = self.fc1(gate)
        x = self.relu(x)
        x = self.fc1(x)

        y = self.fc1(gate)
        y = self.relu(x)
        y = self.fc2(x)
        return x, y
    
    def training_step(self, batch, batch_idx):
        x, (y_1, y_2) = batch
        y_hat_1, y_hat_2 = self.forward(x)

        criterion = nn.MSELoss()
        loss = criterion(y_hat_1, y_1) + criterion(y_hat_2, y_2)
        tensorboard_logs = {'train_loss': loss}
        self.log(f'train loss: {loss}')
        return {'loss': loss, 'log': tensorboard_logs}

    def MAPELoss(self, output, target):
        print(f'target: {target.shape}\toutput: {output.view(-1).shape}')
        print(output)
        exit()
        return torch.mean(torch.abs((target - output.view(-1)) / target))  
    
    def validation_step(self, batch, batch_idx):
        x, (y_1, y_2) = batch
        y_hat_1, y_hat_2 = self.forward(x)
        criterion = self.MAPELoss
        mape_loss = criterion(y_hat_1, y_1) + criterion(y_hat_2, y_2)

        tensorboard_logs = {'MAPE:':mape_loss}
        return {'val_loss': mape_loss, 'log':tensorboard_logs}
    
    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        tensorboard_logs = {'val_loss': avg_loss}
        return {'val_loss': avg_loss, 'log': tensorboard_logs}
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
    
    def collate_fn(self, batch):
        for sample in batch:  #no empty data
            assert sample[1][0] is not None
        
        out = [sample for sample in batch if sample[0] is not None and (sample[1][0] or sample[1][1])]
        return torch.utils.data.dataloader.default_collate(out)

    def train_dataloader(self):
        #Making A dataloader from the fist 4 hvecs
        train_ds = nc.SafeDataset(CalibrationImageDataset('/content/calib-challenge-attempt/', files=[0,1,2,3]))
        train_dataloader = torch.utils.data.DataLoader(train_ds, batch_size=self.batch_size, shuffle=True, num_workers=0, collate_fn=self.collate_fn) #Making A dataloader from the fist 4 hvecs
        return train_dataloader
    
    def val_dataloader(self):
      #Making A dataloader from the last file
      val_ds = nc.SafeDataset(CalibrationImageDataset('/content/calib-challenge-attempt/', files=[4]))
      val_dataloader = torch.utils.data.DataLoader(val_ds, batch_size=self.batch_size,  num_workers=0, collate_fn=self.collate_fn)
      return val_dataloader