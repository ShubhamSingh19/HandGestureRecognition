
import matplotlib
matplotlib.use("Agg")

from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.preprocessing.image import img_to_array
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from pyimagesearch.smallervggnet import SmallerVGGNet
import matplotlib.pyplot as plt
from imutils import paths
import numpy as np
import argparse
import random
import pickle
import cv2
import os

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
	help="dataset")
ap.add_argument("-m", "--model", required=True,
	help="pokedex.model")
ap.add_argument("-l", "--labelbin", required=True,
	help="lb.pickle")
ap.add_argument("-p", "--plot", type=str, default="plot.png",
	help="plot")
args = vars(ap.parse_args())

EPOCHS = 130
INIT_LR = 1e-5
BS = 32
IMAGE_DIMS = (160, 160, 1)

data = []
labels = []

print("[INFO] loading images...")
imagePaths = sorted(list(paths.list_images(args["dataset"])))
random.seed(42)
random.shuffle(imagePaths)


for imagePath in imagePaths:
	img1 = cv2.imread(imagePath,1)
	img1 = cv2.resize(img1,(160,160))
	hsv = cv2.cvtColor(img1,cv2.COLOR_BGR2HSV)
	low = np.array([0,40,30])
	high = np.array([43,255,254])
	image_mask = cv2.inRange(hsv, low, high)
	output = cv2.bitwise_and(img1, img1, mask = image_mask)
	L1 = cv2.Canny(output,50,280,L2gradient=False)
	image = img_to_array(L1)
	data.append(image)
 
	label = imagePath.split(os.path.sep)[-2]
	labels.append(label)

data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)
print("[INFO] data matrix: {:.2f}MB".format(
	data.nbytes / (1024 * 1000.0)))

lb = LabelBinarizer()
labels = lb.fit_transform(labels)

(trainX, testX, trainY, testY) = train_test_split(data,
	labels, test_size=0.3, random_state=42)

aug = ImageDataGenerator(rotation_range=25, width_shift_range=0.1,
	height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
	horizontal_flip=True, fill_mode="nearest")

print("[INFO] compiling model...")
model = SmallerVGGNet.build(width=IMAGE_DIMS[1], height=IMAGE_DIMS[0],
	depth=IMAGE_DIMS[2], classes=len(lb.classes_))
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss="categorical_crossentropy", optimizer=opt,
	metrics=["accuracy"])

print("[INFO] training network...")
H = model.fit_generator(
	aug.flow(trainX, trainY, batch_size=BS),
	validation_data=(testX, testY),
	steps_per_epoch=len(trainX) // BS,
	epochs=EPOCHS, verbose=1)

print("[INFO] serializing network...")
model.save(args["model"])

print("[INFO] serializing label binarizer...")
f = open(args["labelbin"], "wb")
f.write(pickle.dumps(lb))
f.close()

plt.style.use("ggplot")
plt.figure()
N = EPOCHS
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")

plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
#plt.figure()
plt.plot(np.arange(0, N), H.history["acc"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="upper left")
plt.savefig(args["plot"])
