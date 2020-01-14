from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.utils import np_utils
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import pickle
import cv2
import os
import os
import numpy as np
import cv2
import pandas as pd
from sklearn import preprocessing
import sys
import time
import warnings

# CNN
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.constraints import maxnorm
from keras.optimizers import SGD
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization

# Saving and loading model and weights
from keras.models import model_from_json
from keras.models import load_model

# Data preprocessing
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split, StratifiedKFold
from keras.preprocessing.image import ImageDataGenerator

# Read the data from folders
data = []
labels = []

location = './images/detection'  # Root folder
directory = os.listdir(location)
directory = sorted(directory)  # Arrange the data
for i in directory:
    path = location + '/' + i
    sub_dir = os.listdir(path)
    for j in sub_dir:
        # Add the first picture and label into arrays
        image = cv2.imread((path + '/' + j))
        image = cv2.resize(image, dsize=(32, 32))
        data.append(image)
        labels.append(i)


data = np.array(data, dtype="float") / 255.0

num_classes = 2

le = LabelEncoder()
labels = le.fit_transform(labels)
labels = np_utils.to_categorical(labels, 2)

X_train, X_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.15)

# construct the training image generator for data augmentation
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.15,
    shear_range=0.15,
    horizontal_flip=True,
    fill_mode="nearest")


model = Sequential()
model.add(
    Conv2D(
        16, (3, 3), padding="same", input_shape=(
            32, 32, 3), activation='relu'))
model.add(BatchNormalization(axis=-1))
model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(BatchNormalization(axis=-1))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(32, (3, 3), padding="same", activation='relu'))
model.add(BatchNormalization(axis=-1))
model.add(Conv2D(32, (3, 3), padding="same", activation='relu'))
model.add(BatchNormalization(axis=-1))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))


# Compile model
epochs = 50
lrate = 0.0001
decay = lrate / epochs
sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)

model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
print(model.summary())


model.fit_generator(
    datagen.flow(
        X_train,
        y_train,
        batch_size=8),
    validation_data=(
        X_test,
        y_test),
    steps_per_epoch=len(X_train) //
    8,
    epochs=epochs)


'''----------------------------------------------------------------------------------------------'''
# Accuracy of model
result = model.evaluate(X_test, y_test, verbose=0)
print("Test_Acc: %.2f%%" % (result[1] * 100))
'''----------------------------------------------------------------------------------------------'''

# To json file
cv_to_json = model.to_json()
with open("./output_detect/model.json", "w") as file:
    file.write(cv_to_json)
# To HDF5 file
model.save_weights("./output_detect/model.h5")
print("Saved model to disk")
'''----------------------------------------------------------------------------------------------'''
#################### END ####################
