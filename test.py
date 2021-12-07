from calib_dataset import *

tf_train_ds = TFCalibrationDataset('/content/calib-challenge-attempt/', files=[0,1,4,3], batch_size=2)
tf_train_ds = SequenceGenerator('/content/calib-challenge-attempt/', files=[0,1,4,3], batch_size=2)

print('Batches:', len(tf_train_ds))

for sequence in range(0,10):
    print(tf_train_ds.__getitem__(sequence))