# imports
import torch
import cv2
import numpy as np
import random
import glob

class CalibrationImageDataset(torch.utils.data.IterableDataset): #torch.utils.data.IterableDataset
    def __init__(self, root_folder, files:list):  # ./
        self.root_folder = root_folder
        self.target = []
        self.files = files #list of all the files needed

    def __len__(self):
        temp = []
        for i in self.files:
            temp.append(len(glob.glob(f'{self.root_folder}data_{i}/*')))
        return sum(temp)

    def preprocess(self, image):
        # split image into 2 parts, just covering the hood of the Car
        img = cv2.cvtColor(cv2.imread(image), cv2.COLOR_BGR2GRAY)
        height, width, channels = img.shape + (1,)
        croppedImage = img[int(height/2)+100:height, 0:width] #this line crops
        #(thresh, blackAndWhiteImage) = cv2.threshold(img, 35, 175, cv2.THRESH_BINARY_INV)
        return croppedImage / 255


    def get_target(self, img_path, video_num) -> list:
        target_file = open(f"/content/calib-challenge-attempt/calib_challenge/labeled/{video_num}.txt", "r")
        target_idx = int(img_path.split("/")[-1].split(".")[0])

        # take target_idx'th line number and convert targets to a list
        with target_file as file:
            target_pair = file.readlines()[target_idx]

        return [np.float64(num.replace("\n", "")) for num in target_pair.split()]

    def __iter__(self):
        random.shuffle(self.files)

        for video_num in self.files:
            src = glob.glob(f"{self.root_folder}data_{video_num}/*.jpg")
            random.shuffle(src)

            for image_path in src:
                if not np.all(np.isnan(self.get_target(image_path, video_num))):
                    yield self.preprocess(image_path), self.get_target(image_path, video_num)