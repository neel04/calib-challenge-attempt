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
class SequenceGenerator(tf.keras.utils.Sequence):

    def __init__(self, root_folder, files:list, batch_size, scalar:int):  # ./
        self.root_folder = root_folder
        self.target = []
        self.files = files #list of all the files needed
        self.src = sum([glob.glob(f"{self.root_folder}data_{video_num}/*.jpg") for video_num in self.files], []) #converting nested list to flat
        self.batch_size = batch_size
        self.scalar = scalar

    def __len__(self):
        return int(np.ceil(len(self.src) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_img = self.src[idx * self.batch_size:(idx + 1) * self.batch_size]
        returned_images = torch.Tensor(np.array([self.preprocess(image) for image in batch_img])).numpy()
        tgts = np.array([self.get_target(image, image.split('/')[-2].split('_')[1]) for image in batch_img])
        if not any(x is None for x in tgts):
          return returned_images, tgts

    def preprocess(self, image):
        # split image into 2 parts, just covering the hood of the Car
        img = cv2.cvtColor(cv2.imread(image), cv2.COLOR_BGR2RGB)
        height, width, _ = img.shape
        #print(f'h: {height}\tw: {width}')
        croppedImage = img[int(height/2)+200:height, 0:width] #this line crops
        #croppedImage = cv2.resize(croppedImage, (512, 256))
        #(thresh, blackAndWhiteImage) = cv2.threshold(img, 35, 175, cv2.THRESH_BINARY_INV)
        #print(f'final_shape: {croppedImage.shape}')
        return croppedImage / 255

    def get_target(self, img_path, video_num) -> list:
        target_file = open(f"/content/calib-challenge-attempt/calib_challenge/labeled/{video_num}.txt", "r")
        target_idx = int(img_path.split("/")[-1].split(".")[0])

        # take target_idx'th line number and convert targets to a list
        with target_file as file:
            target_pair = file.readlines()[target_idx]
        
        tgt = np.array([np.float32(num.replace("\n", "")) for num in target_pair.split()])
        if not np.isnan(np.sum(tgt)) and not any(x is None for x in tgt):
          return tgt * self.scalar
