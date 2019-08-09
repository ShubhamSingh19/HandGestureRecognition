
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import argparse
import imutils
import pickle
import cv2
import os
import copy

import operator
dicto = {}
dicto["A"] = 0
dicto["B"] = 0
dicto["C"] = 0
dicto["G"] = 0
dicto["O"] = 0
dicto["P"] = 0
dicto["Q"] = 0
dicto["V"] = 0
dicto["W"] = 0
dicto["Y"] = 0
count = [0]

class Classify:
     @staticmethod
     def outputclass(InputImage,Model,Lb):
          image = InputImage
          cv2.imshow("received image",image)
          image = cv2.resize(image, (160, 160))
          L1 = cv2.Canny(image,50,280,L2gradient=False)
          cv2.imshow("canny",L1)
          image = L1.astype("float") / 255.0
          image = img_to_array(image)
          image = np.expand_dims(image, axis=0)

          model = Model
          lb = Lb
          
          
          proba = model.predict(image)[0]
          idx = np.argmax(proba)
          label = lb.classes_[idx]
          dicto[str(label)] = dicto[str(label)]+1
          count[0] = count[0]+1
          if(count[0]==20):
               print("[INFO] classifying image...")
               print(max(dicto.keys(),key = (lambda k: dicto[k])))
               count[0] = 0
               dicto["A"] = 0
               dicto["B"] = 0
               dicto["C"] = 0
               dicto["G"] = 0
               dicto["O"] = 0
               dicto["P"] = 0
               dicto["Q"] = 0
               dicto["V"] = 0
               dicto["W"] = 0
               dicto["Y"] = 0
          return label
