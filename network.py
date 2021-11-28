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
        self.fc1 = nn.Linear(4096, self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size, self.output_size)
        self.relu = torch.nn.ReLU()
        self.maxpool1 = torch.nn.MaxPool2d(stride=4, kernel_size=3)
        self.maxpool2 = torch.nn.MaxPool2d(stride=4, kernel_size=3)
        self.maxpool3 = torch.nn.MaxPool2d(stride=4, kernel_size=3)

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

        gate = x.view(x.size(0), -1)

        z = self.fc1(gate)
        z = self.relu(z)
        z = self.fc2(z)

        y = self.fc1(gate)
        y = self.relu(y)
        y = self.fc2(y)
        return z, y
    
    def training_step(self, batch, batch_idx):
        x, (y_1, y_2) = batch
        y_hat_1, y_hat_2 = self.forward(x)

        criterion = nn.MSELoss()
        loss = criterion(y_hat_1.view(-1).float(), y_1.float()) + criterion(y_hat_2.view(-1).float(), y_2.float())
        tensorboard_logs = {'train_loss': loss}
        self.log('train loss', loss, on_step=True)                #has to log a key-value pair ==> 'train loss:', loss
        return {'loss': loss, 'log': tensorboard_logs}

    def MAPEMetric(self, output, target):
        print(f'\noutput: {output.view(-1)}\ntarget: {target}')
        return torch.mean(torch.abs((target.view(-1) - output.view(-1)) / target)) * 100
    
    def validation_step(self, batch, batch_idx):
        x, (y_1, y_2) = batch
        y_hat_1, y_hat_2 = self.forward(x)
        criterion = self.MAPEMetric
        mape_loss = criterion(y_hat_1.view(-1).float(), y_1.float()) + criterion(y_hat_2.view(-1).float(), y_2.float())

        tensorboard_logs = {'MAPE:':mape_loss}
        self.log('validation_MAPE', tensorboard_logs, on_step=True)
        return {'val_loss': mape_loss.detach(), 'log':tensorboard_logs}
    
    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        tensorboard_logs = {'val_loss': avg_loss}
        self.log('Epoch_End_validation', tensorboard_logs)
        return {'val_loss': avg_loss, 'log': tensorboard_logs}
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def train_dataloader(self):
        #Making A dataloader from the fist 4 hvecs
        train_ds = nc.SafeDataset(CalibrationImageDataset('/content/calib-challenge-attempt/', files=[0,1,4,3]))
        train_dataloader = torch.utils.data.DataLoader(train_ds, batch_size=self.batch_size, shuffle=True, num_workers=4) #Making A dataloader from the fist 4 hvecs
        return train_dataloader
    
    def val_dataloader(self):
      #Making A dataloader from the last file
      val_ds = nc.SafeDataset(CalibrationImageDataset('/content/calib-challenge-attempt/', files=[2]))  #2 is slightly different, hence good test for generalization
      val_dataloader = torch.utils.data.DataLoader(val_ds, batch_size=self.batch_size,  num_workers=4)
      return val_dataloader