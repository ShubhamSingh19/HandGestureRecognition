import matplotlib
matplotlib.use("Agg")

from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
import matplotlib.pyplot as plt
from imutils import paths
import numpy as np
import argparse
import random
import os

data = []
imagePaths = sorted(list(paths.list_images("dataset/W")))
for imagePath in imagePaths:
     img1 = load_img(imagePath,target_size=(160,200))
     image = img_to_array(img1)
     data.append(image)
     
data = np.array(data, dtype="float")

datagen = ImageDataGenerator(brightness_range=[1.2,1.8])

datagen.fit(data)

os.makedirs('imagesw')
ct=0
for X in datagen.flow(data,save_to_dir='imagesw',save_prefix='aug',save_format='jpeg'):
     if ct>=5:
          break
     ct=ct+1
