from classify1 import Classify

import cv2
import imutils
import numpy as np
from keras.models import load_model
import pickle

print("[INFO] loading network...")
model = load_model("pokedex.model")
lb = pickle.loads(open("lb.pickle", "rb").read())

if __name__ == "__main__":

     camera = cv2.VideoCapture(0)
     top, right, bottom, left = 40,240,150,380
     while camera.isOpened():
          keypress = cv2.waitKey(1) & 0xFF
          (ret, frame) = camera.read()
          frame = imutils.resize(frame, width=400)
          frame = cv2.flip(frame, 1)
          roi = frame[top:bottom, right:left]
          cv2.rectangle(frame, (left, top), (right, bottom), (0,255,0), 2)
          
          #if keypress == ord("a"):
          label = Classify.outputclass(roi,model,lb)
          #cv2.putText(frame, label, (100, 100), cv2.FONT_HERSHEY_SIMPLEX,2.7, (0, 255, 0))

          cv2.imshow("Video Feed", frame)

          if keypress == ord("q"):
               break
          
camera.release()
cv2.destroyAllWindows()
