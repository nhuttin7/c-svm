# Hello, I'm Nguyen Nhut Tin.

'''----------------------------------------------------------------------------------------------'''
# Lib
# System and Computer Vision platform
from imutils.video import VideoStream
from imutils import face_utils
from imutils.face_utils import FaceAligner
import imutils
import time
import cv2
import os
import numpy as np
import sys
import warnings
import json
import tensorflow as tf

# Saving and loading model and weights
from keras.models import model_from_json
from keras.models import load_model
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input

# Json tool
import jsonpickle

from joblib import load

# END-LIB
'''----------------------------------------------------------------------------------------------'''


def PrepareImageForPredict(face, h, w):
    face = cv2.resize(face, (h, w))
    face = face.astype("float") / 255.0
    face = img_to_array(face)
    face = np.expand_dims(face, axis=0)
    return face


'''----------------------------------------------------------------------------------------------'''
net = cv2.dnn.readNetFromCaffe(
    "./lib/deploy.prototxt",
    "./lib/res10_300x300_ssd_iter_140000.caffemodel")

# load json and model
json_file_detect = open('./output_detect/model.json', 'r')

model_json_detect = json_file_detect.read()
json_file_detect.close()

global model_detect
model_detect = model_from_json(model_json_detect)
global model_recognize
global graph
graph = tf.get_default_graph()


# load weights into new model
model_detect.load_weights("./output_detect/model.h5")
model_recognize = load('./output_recognize/model.joblib') 
print("Loaded model from disk")


labels_detect = ["yes", "no"]
model = VGG16(weights='imagenet', include_top=False)

video_capture = cv2.VideoCapture(0)

while True:
    if (video_capture.isOpened()== False):
        print ("Not opening camera")
        continue


    # Capture frame-by-frame
    _, frame = video_capture.read()

    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(
        frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            startX, startY = max(0, startX), max(0, startY)
            endX, endY = min(w, endX), min(h, endY)

            face = frame[startY:endY, startX:endX]
            if np.array(face).shape[1] != 0 and np.array(face).shape[0] != 0:

                face1 = PrepareImageForPredict(face, 32, 32)
                pred = model_detect.predict(face1)[0]
                j = np.argmax(pred)

                if labels_detect[j] == "yes":
                    face2 = PrepareImageForPredict(face, 224, 224)
                    img_data = preprocess_input(face2)

                    # Extract feature from image and flatten it into n-dimensional 
                    vgg16_feature = model.predict(img_data)
                    features = np.array(vgg16_feature).flatten()
                    features = np.expand_dims(features, axis=0)
                    label = model_recognize.predict(features)
                    if len(label) == 1:
                        label = label[0]
                    cv2.rectangle(
                        frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
                    cv2.putText(frame, label, (startX, startY - 10),
                                1, 0.5, (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('Video', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
