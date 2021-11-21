# imports
import os
import torch
from torchvision.io import read_image
import numpy as np
import matplotlib.pyplot as plt
import glob


class CalibrationImageDataset(torch.utils.data.IterableDataset):
    def __init__(self, root_folder):  # ./
        self.root_folder = root_folder
        self.target = []
        self.len = 2000

    def __len__(self):
        return self.len

    def preprocess(self, image):
        # split image into 2 parts, just covering the hood of the Car
        image = read_image(image)
        print(image.shape)
        return image


    def get_target(self, img_path, video_num) -> list:
        target_file = open(f"./calib_challenge/labeled/{video_num}.txt", "r")
        target_idx = int(img_path.split("/")[-1].split(".")[0])

        # take target_idx'th line number and convert targets to a list
        with target_file as file:
            target_pair = file.readlines()[target_idx]

        return [np.float64(num.replace("\n", "")) for num in target_pair.split()]

    def __iter__(self):
        for video_num in range(0, 5):
            for image_path in glob.glob(f"{self.root_folder}data_{video_num}/*.jpg"):
                if not np.all(np.isnan(self.get_target(image_path, video_num))):
                    yield self.preprocess(image_path), self.get_target(image_path, video_num)