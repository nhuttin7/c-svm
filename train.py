from keras.preprocessing import image
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
import numpy as np
import csv
import cv2

from sklearn import svm
import os

from joblib import dump



def PreprocessInput(location):
    # Using pre-trained VGG16 model to extract feature
    model = VGG16(weights='imagenet', include_top=False)

    X = []
    y = []
    directory = os.listdir(location)
    directory = sorted(directory)  # Arrange the data
    for i in directory:
        path = location+"/"+i
        img = image.load_img(path, target_size=(224, 224))
        img_data = image.img_to_array(img)
        img_data = np.expand_dims(img_data, axis=0)
        img_data = preprocess_input(img_data)

        # Extract feature from image and flatten it into n-dimensional 
        vgg16_feature = model.predict(img_data)
        features = np.array(vgg16_feature).flatten()

        X.append(features)
        y.append(i)
    
    return X, y

path = "./images"

X, y = PreprocessInput(path)

clf = svm.SVC()
clf.fit(X, y)


dump(clf, './output_recognize/model.joblib',protocol=2) 