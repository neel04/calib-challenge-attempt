from calib_dataset import SequenceGenerator
from hvec import execute_shell, hevc_to_frames

import tensorflow as tf
import wandb
from wandb.keras import WandbCallback
from tqdm import tqdm
import os
import numpy as np
import autokeras as ak

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
BATCH_SIZE = 16
tf_train_ds = SequenceGenerator('/content/calib-challenge-attempt/', files=[0,1,4,3], batch_size=BATCH_SIZE)
tf_val_ds = SequenceGenerator('/content/calib-challenge-attempt/', files=[2], batch_size=BATCH_SIZE)

# Initialize the multi with multiple inputs and outputs.
def MAPEMetric(target, output):
        return tf.math.reduce_mean(tf.math.abs((output - target) / output)) * 100

model = ak.ImageRegressor(
    output_dim=2,
    loss="mean_squared_error",
    metrics=[MAPEMetric],
    project_name="image_regressor",
    max_trials=150,
    objective="val_loss",
    overwrite=False,
    directory='/content/drive/MyDrive/Comma_AI/',    #Directory to sync progress @ cloud
    seed=69420
    )
'''
model = ak.AutoModel(
    inputs=[ak.ImageInput()],
    outputs=[
        ak.RegressionHead(metrics=["mae"], output_dim=2),
        ],
    overwrite=True,
    max_trials=1,
    seed=69420
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
            
print(f'Produced sample shape: {tf_train_ds.__getitem__(2)[0].shape}')

training =  tf.data.Dataset.from_generator(train_gen_data_generator, output_signature=(
        tf.TensorSpec(shape=(None, 256, 512), dtype=tf.float32),
        tf.TensorSpec(shape=(None, 2), dtype=tf.float32)))

validation =  tf.data.Dataset.from_generator(val_gen_data_generator, output_signature=(
        tf.TensorSpec(shape=(None, 256, 512), dtype=tf.float32),
        tf.TensorSpec(shape=(None, 2), dtype=tf.float32)))

#WANDB logging
wandb.init(project="CalibNet", 
           name="NAS_Run",
           config={"hyper":"parameters"})


model.fit(x=training, validation_data=validation, epochs=3, batch_size=BATCH_SIZE, shuffle=True, callbacks=[WandbCallback()])
