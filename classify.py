
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import argparse
import imutils
import pickle
import cv2
import os
import copy

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="examples")
args = vars(ap.parse_args())

image = cv2.imread(args["image"])

 
image = cv2.resize(image, (160, 160))
L1 = cv2.Canny(image,50,280,L2gradient=False)
image = L1.astype("float") / 255.0
image = img_to_array(image)
image = np.expand_dims(image, axis=0)

print("[INFO] loading network...")
model = load_model("pokedex.model")
lb = pickle.loads(open("lb.pickle", "rb").read())

print("[INFO] classifying image...")
proba = model.predict(image)[0]
idx = np.argmax(proba)
label = lb.classes_[idx]
'''
output = imutils.resize(output, width=400)
cv2.putText(output, label, (10, 25),  cv2.FONT_HERSHEY_SIMPLEX,
	0.7, (0, 255, 0), 2)

# show the output image
print("[INFO] {}".format(label))
cv2.imshow("Output", output)
cv2.waitKey(0)
'''

print(model.layer[0].output)
