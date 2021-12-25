import os
import cv2
from tqdm.notebook import tqdm
from pathlib import Path
import numpy as np

def execute_shell(command):
    os.system(command)

def hevc_to_frames(sample_num, out_folder):
  if not Path(f'./{out_folder}').exists():
    execute_shell(f'mkdir {out_folder}')

  vidcap = cv2.VideoCapture(f'/content/calib-challenge-attempt/calib_challenge/labeled/{sample_num}.hevc')   # sample_num --> int

  success, image = vidcap.read()
  
  count = 0
  
  while success:
    success, image = vidcap.read()

    if not success:
      break
      
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(f"./{out_folder}/{count}.jpg", image)     # save frame as JPEG file
    count += 1
  
  #print(f'{sample_num} has been completed')

def nan_i_nator(base_path, files):
  for file in files:
    lines = open(f'{base_path}/{file}.txt').readlines()
    idxes = []
    cleaned = []

    for location, line in enumerate(lines):
      if "nan" in line:
        idxes.append(location)
      else:
        cleaned.append(line)

    with open(f'{base_path}/{file}.txt', 'w') as outfile:
      outfile.write("\n".join(str(item) for item in cleaned))

    print(f'cleaned {len(idxes} NaNs in {file}!')
    
    [execute_shell(f'rm /content/calib-challenge-attempt/data_{file}/{i}.jpg') for i in idxes]
