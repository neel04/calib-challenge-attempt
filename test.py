from calib_dataset import TFCalibrationDataset

tf_train_ds = TFCalibrationDataset('/content/calib-challenge-attempt/', files=[0,1,4,3], batch_size=2)
print('Batches:', len(tf_train_ds))

for sequence in tf_train_ds:
    print(sequence[1].shape)
    exit()