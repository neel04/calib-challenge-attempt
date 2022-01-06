from calib_dataset import SequenceGenerator
from hvec import test_hevc_to_frames

import autokeras as ak
import tensorflow as tf
import wandb
import glob
import tqdm
import cv2
import numpy as np
import os

def MAPEMetric(target, output):
        return tf.math.reduce_mean(tf.math.abs((output - target) / output)) * 100

loaded_model = tf.keras.models.load_model("/content/drive/MyDrive/Comma_AI/checkpoints/", custom_objects={'MAPEMetric': MAPEMetric})
loaded_model.compile(loss='mean_absolute_error', metrics=[MAPEMetric])
loaded_model.summary()

tf_val_ds = SequenceGenerator('/content/calib-challenge-attempt/', files=[2], batch_size=32, scalar=10000)

def val_gen_data_generator():
    for i in range(len(tf_val_ds)):
        if tf_val_ds.__getitem__(i) is not None:
            yield tf_val_ds.__getitem__(i)

validation =  tf.data.Dataset.from_generator(val_gen_data_generator, output_signature=(
        tf.TensorSpec(shape=(None, 337, 582), dtype=tf.float32),
        tf.TensorSpec(shape=(None, 2), dtype=tf.float32))) #.prefetch(tf.data.AUTOTUNE)

print(f'\nvalidation: {validation}\n')
predicted_y = loaded_model.evaluate(validation)
print(predicted_y)

#----------------------------
#Setting up test datasets
if not os.path.exists('/content/calib-challenge-attempt/test_*'):
  for i in range(5,10):
    test_hevc_to_frames(i, f'./test_{i}')
  print('\nTest Dataset constructed!')   #Constructs test dataset
#----------------------------

sample_pred = loaded_model.predict(tf.ones((1, 337, 582)))
print(f'sample predictions: {sample_pred}')

def preprocess(image):
  # split image into 2 parts, just covering the hood of the Car
  img = cv2.cvtColor(cv2.imread(image), cv2.COLOR_BGR2RGB)
  height, width, _ = img.shape
  #print(f'h: {height}\tw: {width}')
  croppedImage = img[int(height/2)+100:height, 0:width] #this line crops
  #Add a red bar strategically for no fucking reason
  img = cv2.resize(croppedImage, None, fx=0.5, fy=1, interpolation=cv2.INTER_AREA)
  img = cv2.line(img, (0,80), (582, 80), (0,0,255), 12)
  return img[:, :, 0] / 255 # ==> B&W

#Compute the predictions
for index in tqdm.tqdm(range(5,10)):
  files = glob.glob(f'/content/calib-challenge-attempt/test_{index}/*.jpg')
  results = []

  for image in files:
    processed_img = np.expand_dims(preprocess(image), axis=0)
    preds = loaded_model.predict(processed_img) / tf_val_ds.scalar
    results.extend(preds.tolist())
  
  with open(f'./{index}.txt', 'w') as out_file:
    results = [' '.join(str(_) for _ in k) for k in results]
    [out_file.write(str(i+'\n')) for i in results]
