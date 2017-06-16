# Generic imports
import csv
import cv2
import random
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from scipy.misc import imread, imsave
import tensorflow as tf

# Keras imports
import keras
from keras.optimizers import *
from keras.models import Sequential, model_from_json, load_model
from keras.layers import Flatten, Dense, Activation, Dropout, Lambda, Cropping2D, ELU
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.core import Dropout
from keras.callbacks import EarlyStopping

# File Management
FOLDER_PATH = 'data/'
FILE_CSV = './data/driving_log.csv'
FILE_MODEL_H5 = 'model.h5'
FILE_MODEL_JSON = 'model.json'

# Load CSV for 4 Columns
DATA = pd.read_csv(FILE_CSV, usecols = range(0,4)) 

## Split the training data into training and validation
TRAIN_DATA, VALIDATION_DATA = train_test_split(DATA, test_size = 0.15)

# Hyper-parameters Settings
BATCH_SIZE = 32
NUMBER_OF_EPOCHS = 10
ACTIVATION = 'relu'
NUM_TRAIN_DATA, NUM_VALID_DATA = len(TRAIN_DATA), len(VALIDATION_DATA)

def generator(data, batch_size):
    POSITION, CORRECTION, DATA_SIZE = ['left', 'center', 'right'], [.25, 0, -.25], len(data)
    while True:
        for start in range(0, DATA_SIZE, batch_size):
            images, measurements = [], []
            
            # Reading images and measurement for 3 angles which is ['left', 'center', 'right']
            for i in range(3):
                for rows in range(start, start + batch_size):
                    if rows < DATA_SIZE:
                        row = data.index[rows]
                        measurement = data['steering'][row] + CORRECTION[i] # create adjusted steering measurements for the side camera images
                        #if measurement != 0:
                        image = imread(FOLDER_PATH + data[POSITION[i]][row].strip()) # Reading images and remove whitespace in image path
                        measurements.extend([measurement, -measurement]) # image, flipped image
                        images.extend([image, np.fliplr(image)])  # image, flipped image
            yield np.array(images), np.array(measurements)
            
            
# Compile and train the model using the generator function
train_generator = generator(TRAIN_DATA, batch_size=BATCH_SIZE)
validation_generator = generator(VALIDATION_DATA, batch_size=BATCH_SIZE)

# Building the model according to Nvidia's Model from https://arxiv.org/pdf/1604.07316v1.pdf
model = Sequential()
model.add(Lambda(lambda x: (x / 127.5) - 1., input_shape = (160, 320, 3)))
model.add(Cropping2D(cropping=((74,24), (0,0))))
model.add(Convolution2D(24,5,5,subsample=(2,2),activation=ACTIVATION))
model.add(Convolution2D(36,5,5,subsample=(2,2),activation=ACTIVATION))
model.add(Convolution2D(48,5,5,subsample=(2,2),activation=ACTIVATION))
model.add(Convolution2D(64,3,3,activation=ACTIVATION))
model.add(Convolution2D(64,3,3,activation=ACTIVATION))
model.add(Flatten())
model.add(Dropout(.5))
model.add(Dense(1164))
model.add(Activation(ACTIVATION))
model.add(Dense(100))
model.add(Activation(ACTIVATION))
model.add(Dense(50))
model.add(Activation(ACTIVATION))
model.add(Dense(10))
model.add(Activation(ACTIVATION))
model.add(Dense(1))
model.summary()
model.compile(loss='mse', optimizer='adam')

# Stop training when a monitored quantity has stopped improving for 2 epochs
# early_stopping = EarlyStopping(monitor='val_loss', patience = 2, verbose = 1, mode = 'auto')

history_object = model.fit_generator(train_generator,
                 samples_per_epoch = NUM_TRAIN_DATA,
                 validation_data = validation_generator,
                 nb_val_samples = NUM_VALID_DATA,
                 nb_epoch = NUMBER_OF_EPOCHS,
                 #callbacks = [early_stopping],
                 verbose = 1)

print('Saving model...')
model.save(FILE_MODEL_H5)
with open(FILE_MODEL_JSON, "w") as json_file:
    json_file.write(model.to_json())
print("FILE_MODEL.")


"""
### print the keys contained in the history object
print(history_object.history.keys())

### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()
"""

