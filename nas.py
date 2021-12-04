from calib_dataset import CalibrationImageDataset
from hvec import execute_shell, hevc_to_frames
from trains import Task

import tensorflow as tf
import torch
import os
import numpy as np
from tqdm import tqdm
import threading
import nonechucks as nc
import autokeras as ak
import tensorflow_datasets as tfds
import more_itertools as mit

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

train_ds = nc.SafeDataset(CalibrationImageDataset('/content/calib-challenge-attempt/', files=[0,1,4,3])) #Making A dataset from the fist 4 hvecs
val_ds = nc.SafeDataset(CalibrationImageDataset('/content/calib-challenge-attempt/', files=[2]))  #2 is slightly different, hence good test for generalization

class threadsafe_iter:
    """Takes an iterator/generator and makes it thread-safe by
    serializing call to the `next` method of given iterator/generator.
    """
    def __init__(self, it):
        self.it = it
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def __next__(self):
        with self.lock:
            return self.it.__next__()

def threadsafe_generator(f):
    """A decorator that takes a generator function and makes it thread-safe.
    """
    def g(*a, **kw):
        return threadsafe_iter(f(*a, **kw))
    return g

@threadsafe_generator
def data_generator(batch_size, dataset):
    '''
    Dataset Generator for easily constructing tf.Dataset;
    adapter for a torch Dataset.
    
    returns (batch_sizes, 256, 512) + (batch_size, 2)
    '''
    dloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=4, drop_last=True)

    for img, tgt in iter(dloader):
        tgt = np.rad2deg([i.numpy() for i in tgt]) * 1000      #100 is the scaling factor
        img = img.squeeze().numpy()
        yield img, tgt.squeeze(1)

# Creating the generator
BATCH_SIZE = 4

train_data_gen = data_generator(dataset=train_ds, batch_size=BATCH_SIZE)
val_data_gen = data_generator(dataset=val_ds, batch_size=BATCH_SIZE)

def callable_iterator(generator, expected_batch_size):
    for img_batch, targets_batch in generator:
        if img_batch.shape[0] == expected_batch_size and targets_batch.shape[1] == expected_batch_size:
            yield img_batch, targets_batch

train_dataset = tf.data.Dataset.from_generator(
    lambda: train_data_gen,
    output_types=(tf.float32, tf.float32), 
    output_shapes=((256, 512), (2,))).batch(BATCH_SIZE, drop_remainder=True)

val_dataset = tf.data.Dataset.from_generator(
    lambda: val_data_gen,
    output_types=(tf.float32, tf.float32), 
    output_shapes=((256, 512), (2,))).batch(BATCH_SIZE, drop_remainder=True)

for i,j in train_dataset.as_numpy_iterator():
    print(f'\nTF dataset image shape: {i.shape}\nTF dataset target shape: {j.shape}\n\n')
    break

#=================================================================================================
#Autokeras block

# Initialize the multi with multiple inputs and outputs.
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

def MAPEMetric(target, output):
        return np.mean(np.abs((output.view(-1) - target.view(-1)) / output)) * 100

model = ak.ImageRegressor(
    output_dim=2,
    loss="mean_average_error",
    metrics=[MAPEMetric],
    project_name="image_regressor",
    max_trials=100,
    objective="val_loss",
    overwrite=True,
    seed=69420
    )

# Fit the model with prepared data.
#Convert the TF Dataset to an np.array
#Useless AutoKeras shenanigans
#train_numpy = [list(mit.collapse(i)) for i in tfds.as_numpy(train_dataset)]
#val_numpy = [list(mit.collapse(i)) for i in tfds.as_numpy(val_dataset)]

#Setting up TRAINS logging
#task = Task.init(project_name="CalibNet", task_name="Training CalibNet")

model.fit(x=train_dataset, validation_dataset=val_dataset, epochs=3)
