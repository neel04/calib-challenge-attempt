import autokeras as ak
from calib_dataset import SequenceGenerator
import tensorflow as tf
import wandb
import os

#Downloading the Run
if not os.path.exists('./model-best.h5'):
    api = wandb.Api()
    run = api.run("neel/CalibNet/26z4kpq6")
    run.file("model-best.h5").download()
#----------------------------------------------

def MAPEMetric(target, output):
        return tf.math.reduce_mean(tf.math.abs((output - target) / output)) * 100

loaded_model = tf.keras.models.load_model("/content/model-best.h5", custom_objects={'MAPEMetric': MAPEMetric})
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