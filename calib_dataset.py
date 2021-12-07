# imports
from tensorflow.python.ops.gen_batch_ops import batch
import torch
import tensorflow as tf
import cv2
import numpy as np
import glob

class CalibrationImageDataset(torch.utils.data.Dataset):
    def __init__(self, root_folder, files:list):  # ./
        self.root_folder = root_folder
        self.target = []
        self.files = files #list of all the files needed
        self.src = sum([glob.glob(f"{self.root_folder}data_{video_num}/*.jpg") for video_num in self.files], []) #converting nested list to flat

    def __len__(self):
        return len(self.src)

    def preprocess(self, image):
        # split image into 2 parts, just covering the hood of the Car
        img = cv2.cvtColor(cv2.imread(image), cv2.COLOR_BGR2GRAY)
        height, width = img.shape
        croppedImage = img[int(height/2)+100:height, 0:width] #this line crops
        croppedImage = cv2.resize(croppedImage, (512, 256))
        #(thresh, blackAndWhiteImage) = cv2.threshold(img, 35, 175, cv2.THRESH_BINARY_INV)
        return torch.from_numpy(croppedImage / 255).float() #torch.from_numpy(np.expand_dims((croppedImage / 255), 0)).float()


    def get_target(self, img_path, video_num) -> list:
        target_file = open(f"/content/calib-challenge-attempt/calib_challenge/labeled/{video_num}.txt", "r")
        target_idx = int(img_path.split("/")[-1].split(".")[0])

        # take target_idx'th line number and convert targets to a list
        with target_file as file:
            target_pair = file.readlines()[target_idx]
        
        return [np.float64(num.replace("\n", "")) for num in target_pair.split()]   #Scaling: *10000 

    def __getitem__(self, index):
        image_path = self.src[index]
        video_num = int(image_path.split('/')[3][-1])
        img, tgt = self.preprocess(image_path), self.get_target(image_path, video_num)

        if all(isinstance(e, (int, float)) for e in tgt):
            return img, tgt

# Build the keras Sequence dataset for AutoKeras
class TFCalibrationDataset(tf.keras.utils.Sequence):
    def __init__(self, root_folder, files:list, batch_size=2):  # ./
        self.root_folder = root_folder
        self.target = []
        self.files = files #list of all the files needed
        self.src = sum([glob.glob(f"{self.root_folder}data_{video_num}/*.jpg") for video_num in self.files], []) #converting nested list to flat
        self.batch_size = batch_size

    def __len__(self):
        return len(self.src) // self.batch_size #number of batches in the sequence

    def preprocess(self, image):
        # split image into 2 parts, just covering the hood of the Car
        img = cv2.cvtColor(cv2.imread(image), cv2.COLOR_BGR2GRAY)
        height, width = img.shape
        croppedImage = img[int(height/2)+100:height, 0:width] #this line crops
        croppedImage = cv2.resize(croppedImage, (512, 256))
        #(thresh, blackAndWhiteImage) = cv2.threshold(img, 35, 175, cv2.THRESH_BINARY_INV)
        return tf.convert_to_tensor(croppedImage / 255, dtype=tf.float32) #torch.from_numpy(np.expand_dims((croppedImage / 255), 0)).float()


    def get_target(self, img_path, video_num) -> list:
        target_file = open(f"/content/calib-challenge-attempt/calib_challenge/labeled/{video_num}.txt", "r")
        target_idx = int(img_path.split("/")[-1].split(".")[0])

        # take target_idx'th line number and convert targets to a list
        with target_file as file:
            target_pair = file.readlines()[target_idx]
        
        return [np.float64(num.replace("\n", "")) for num in target_pair.split()]   #Scaling: *10000 

    def __getitem__(self, index):
        batched_image_paths = self.src[index * self.batch_size:(index + 1) * self.batch_size]
        batch_x = [self.preprocess(image_path) for image_path in batched_image_paths]
        batch_y = [self.
        get_target(image_path, int(image_path.split('/')[3][-1])) for image_path in batched_image_paths][index * self.batch_size:(index + 1) *
        self.batch_size]

        # This works fine, check nas.py
        if all(isinstance(e[0], (int, float)) for e in batch_y) and batch_y != []:
            print("Batch y:", batch_y)
            return np.array(batch_x), np.array(batch_y)
    
    def getitem(self, index):
        self.__getitem__(index)


def DatasetFromSequenceClass(sequenceClass, stepsPerEpoch, nEpochs, batchSize, dims=[512,512,3], n_features=2, data_type=tf.float32, label_type=tf.float32):
    # eager execution wrapper
    def DatasetFromSequenceClassEagerContext(func):
        def DatasetFromSequenceClassEagerContextWrapper(batchIndexTensor):
            # Use a tf.py_function to prevent auto-graph from compiling the method
            tensors = tf.py_function(
                func,
                inp=[batchIndexTensor],
                Tout=[data_type, label_type]
            )

            # set the shape of the tensors - assuming channels last
            tensors[0].set_shape([batchSize, dims[0], dims[1], dims[2]])   # [samples, height, width, nChannels]
            tensors[1].set_shape([batchSize, n_features]) # [samples, height, width, nClasses for one hot]
            return tensors
        return DatasetFromSequenceClassEagerContextWrapper

    # TF dataset wrapper that indexes our sequence class
    @DatasetFromSequenceClassEagerContext
    def LoadBatchFromSequenceClass(batchIndexTensor):
        # get our index as numpy value - we can use .numpy() because we have wrapped our function
        batchIndex = batchIndexTensor.numpy()

        # zero-based index for what batch of data to load; i.e. goes to 0 at stepsPerEpoch and starts cound over
        zeroBatch = batchIndex % stepsPerEpoch

        # load data
        data, labels = sequenceClass[zeroBatch]

        # convert to tensors and return
        return tf.convert_to_tensor(data), tf.convert_to_tensor(labels)