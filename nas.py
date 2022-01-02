from calib_dataset import SequenceGenerator
from hvec import execute_shell, hevc_to_frames, nan_i_nator

import tensorflow as tf
import wandb
from wandb.keras import WandbCallback
from kerastuner import Objective
from tqdm import tqdm
import os
import numpy as np
import autokeras as ak

#============================================================================================================
#Removes Nans in files
#nan_i_nator('/content/calib-challenge-attempt/calib_challenge/labeled', [0,1,2,3,4])

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
BATCH_SIZE = 32
tf_train_ds = SequenceGenerator('/content/calib-challenge-attempt/', files=[0,1,4,3], batch_size=BATCH_SIZE, scalar=10000)
tf_val_ds = SequenceGenerator('/content/calib-challenge-attempt/', files=[2], batch_size=BATCH_SIZE, scalar=10000)

# Initialize the multi with multiple inputs and outputs.
def MAPEMetric(target, output):
        return tf.math.reduce_mean(tf.math.abs((output - target) / output)) * 100

input_node = ak.ImageInput()
output_node = ak.Normalization()(input_node)
output_node1 = ak.ConvBlock()(output_node)
#output_node2 = ak.DenseBlock()(output_node1)
#output_node2 = ak.ResNetBlock()(output_node)
#output_node = ak.Merge()([output_node1, output_node2])
output_node = ak.RegressionHead(output_dim=2)(output_node1)

model = ak.AutoModel(
    inputs=input_node, outputs=output_node,
    loss="mean_absolute_error",
    metrics=[MAPEMetric],
    project_name="image_regressor",
    max_trials=25,
    objective="val_loss",
    overwrite=False,
    directory='/kaggle/working/calib-challenge-attempt/image_regressor/',    #Directory to sync progress @ cloud| /content/drive/MyDrive/Comma_AI/
    seed=42
)

'''
model = ak.ImageRegressor(
    output_dim=2,
    loss="mean_squared_error",
    metrics=[MAPEMetric],
    project_name="image_regressor",
    max_trials=100,
    objective="val_loss",
    overwrite=False,
    directory='/kaggle/working/calib-challenge-attempt/',    #Directory to sync progress @ cloud| /content/drive/MyDrive/Comma_AI/
    seed=42
    )
'''
# Fit the model with prepared data.

#Setting up TRAINS logging
#task = Task.init(project_name="CalibNet", task_name="Training CalibNet")

def train_gen_data_generator():
    for i in range(len(tf_train_ds)):
        if tf_train_ds.__getitem__(i) is not None:
            yield tf_train_ds.__getitem__(i)

def val_gen_data_generator():
    for i in range(len(tf_val_ds)):
        if tf_val_ds.__getitem__(i) is not None:
            yield tf_val_ds.__getitem__(i)
            
print(f'Produced sample shape: {tf_train_ds.__getitem__(1)[0].shape}')

training =  tf.data.Dataset.from_generator(train_gen_data_generator, output_signature=(
        tf.TensorSpec(shape=(None, 337, 582), dtype=tf.float32),
        tf.TensorSpec(shape=(None, 2), dtype=tf.float32))).prefetch(tf.data.AUTOTUNE)

validation =  tf.data.Dataset.from_generator(val_gen_data_generator, output_signature=(
        tf.TensorSpec(shape=(None, 337, 582), dtype=tf.float32),
        tf.TensorSpec(shape=(None, 2), dtype=tf.float32))).prefetch(tf.data.AUTOTUNE)

#WANDB logging
wandb.init(project="CalibNet", 
           name="RGB_Run",
           config={"hyper":"parameters"})

for sample in training.as_numpy_iterator():
  result = sample[0].prod()
  if not isinstance(result,np.float32):
    print('\nWarning!\tNaNs found in data. Will lead to errors in gradient flow', type(result))
    raise SystemExit

EStop = tf.keras.callbacks.EarlyStopping(
    monitor='MAPEMetric', min_delta=2, patience=3, verbose=0, mode="min", baseline=100, #baseline is 100
    restore_best_weights=True)

WandbCB = WandbCallback(log_evaluation=True, save_model=True)

model.fit(x=training, validation_data=validation, batch_size=BATCH_SIZE, shuffle=True, callbacks=[WandbCB, EStop])
