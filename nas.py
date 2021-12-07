from calib_dataset import CalibrationImageDataset, TFCalibrationDataset, SequenceGenerator
from hvec import execute_shell, hevc_to_frames

import tensorflow as tf
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

tf_train_ds = SequenceGenerator('/content/calib-challenge-attempt/', files=[0,1,4,3], batch_size=2)
tf_val_ds = SequenceGenerator('/content/calib-challenge-attempt/', files=[2], batch_size=2)


# class threadsafe_iter:
#     """Takes an iterator/generator and makes it thread-safe by
#     serializing call to the `next` method of given iterator/generator.
#     """
#     def __init__(self, it):
#         self.it = it
#         self.lock = threading.Lock()

#     def __iter__(self):
#         return self

#     def __next__(self):
#         with self.lock:
#             return self.it.__next__()

# def threadsafe_generator(f):
#     """A decorator that takes a generator function and makes it thread-safe.
#     """
#     def g(*a, **kw):
#         return threadsafe_iter(f(*a, **kw))
#     return g

# @threadsafe_generator
# def data_generator(batch_size, dataset):
#     '''
#     Dataset Generator for easily constructing tf.Dataset;
#     adapter for a torch Dataset.
    
#     returns (batch_sizes, 256, 512) + (batch_size, 2)
#     '''
#     for idx in range(len(dataset)):
#         img, tgt = dataset[idx]
#         tgt = np.rad2deg([i for i in tgt]) * 1000      #100 is the scaling factor
#         if not np.isnan(tgt[0]):
#           yield img, tgt

# # Creating the generator
# BATCH_SIZE = 4

# train_data_gen = data_generator(dataset=train_ds, batch_size=BATCH_SIZE)
# val_data_gen = data_generator(dataset=val_ds, batch_size=BATCH_SIZE)

# def callable_iterator(generator, expected_batch_size):
#     for img_batch, targets_batch in generator:
#         if img_batch.shape[0] == expected_batch_size:
#             yield img_batch, targets_batch

# train_dataset = tf.data.Dataset.from_generator(
#     lambda: train_data_gen,
#     output_types=(tf.float32, tf.float32), 
#     output_shapes=((256, 512), (2,))
#     ).batch(BATCH_SIZE, drop_remainder=True)

# val_dataset = tf.data.Dataset.from_generator(
#     lambda: val_data_gen,
#     output_types=(tf.float32, tf.float32), 
#     output_shapes=((256, 512), (2,))
#     ).batch(BATCH_SIZE, drop_remainder=True)

# for i,j in train_dataset.as_numpy_iterator():
#     print(f'\nTF dataset image shape: {i.shape}\nTF dataset target shape: {j.shape}\ntarget:{j}\n')
#     break

#=================================================================================================
#Autokeras block

# Initialize the multi with multiple inputs and outputs.
def MAPEMetric(target, output):
        return tf.math.reduce_mean(tf.math.abs((output - target) / output)) * 100

model = ak.ImageRegressor(
    output_dim=2,
    loss="mean_squared_error",
    metrics=[MAPEMetric],
    project_name="image_regressor",
    max_trials=100,
    objective="val_loss",
    overwrite=True,
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
#Convert the TF Dataset to an np.array
#Useless AutoKeras shenanigans
#train_numpy = np.array([list(mit.collapse(i)) for i in tfds.as_numpy(train_dataset)])
#val_numpy = np.array([list(mit.collapse(i)) for i in tfds.as_numpy(val_dataset)])

#Setting up TRAINS logging
#task = Task.init(project_name="CalibNet", task_name="Training CalibNet")
#print('\nSamples going for training:', len([0 for i,j in train_dataset])*BATCH_SIZE)

def gen_data_generator():
    for i in range(len(tf_train_ds)):
        if tf_train_ds.__getitem__(i) is not None:
            yield tf_train_ds.__getitem__(i)

training =  tf.data.Dataset.from_generator(gen_data_generator, output_signature=(
        tf.TensorSpec(shape=(None, 256, 512), dtype=tf.float32),
        tf.TensorSpec(shape=(None, 2), dtype=tf.float32))) 

model.fit(x=training, epochs=3, batch_size=2)