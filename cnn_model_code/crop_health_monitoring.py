## Crop Health Monitoring Project

# Importing Libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

import tensorflow as tf
print("TensorFlow version:", tf.__version__)

import keras
print("Keras version:", keras.__version__)

# Importing Image Data Generator from Keras which is in Tensorflow

from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img

from tensorflow.keras.applications.vgg19 import VGG19, preprocess_input, decode_predictions

## Exploratory Data Analysis

# Checking the number of classes

len(os.listdir("/content/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/train"))

# Checking the names of the classes

os.listdir("/content/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/train")


train_datagen = ImageDataGenerator(zoom_range = 0.5, shear_range = 0.3, horizontal_flip = True, preprocessing_function = preprocess_input)

val_datagen = ImageDataGenerator(preprocessing_function = preprocess_input)

"""## Preprocessing the data"""

train = train_datagen.flow_from_directory(directory = "/content/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/train",
                                          target_size = (256, 256),
                                          batch_size = 32)

val = val_datagen.flow_from_directory(directory = "/content/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/valid",
                                          target_size = (256, 256),
                                          batch_size = 32)

t_img, label = train.next()

t_img

t_img.shape

def plotImage(img_arr, label):
  for img, label in zip(img_arr, label):
    plt.figure(figsize = (5,5))
    # plt.imshow(img)
    plt.imshow(img/255)
    plt.show()

plotImage(t_img[:3], label[:3])

v_img, label = val.next()

plotImage(v_img[:3], label[:3])

"""## Creating the Model"""

from keras.layers import Dense, Flatten
from keras.models import Model
from keras.applications.vgg19 import VGG19
import keras

base_model = VGG19(input_shape = (256, 256, 3), include_top = False)

for layer in base_model.layers:
  layer.trainable = False

base_model.summary()

X = Flatten()(base_model.output) # creating a Flatten layer which takes the output of the base_model as input

X = Dense(units = 38, activation = 'softmax')(X) # takes the output of the Flatten layer, the number of units is equal to the number of classes in our dataset

# creating our model

model = Model(base_model.input, X)

model.summary()

model.compile(optimizer = 'adam', loss = keras.losses.categorical_crossentropy, metrics = ['accuracy'])

"""## Early stopping & Model check point"""

from keras.callbacks import ModelCheckpoint, EarlyStopping

# Early Stopping

es = EarlyStopping(monitor = 'val_accuracy', min_delta = 0.01, patience = 3, verbose = 1)

# Model Check Point

mc = ModelCheckpoint(filepath = 'best_model.h5',
                     monitor = 'val_accuracy',
                     min_delta = 0.01,
                     patience = 3,
                     verbose = 1,
                     save_best_only = True)

# Callback

cb = [es, mc]

his = model.fit_generator(train,
                          steps_per_epoch = 16,
                          epochs = 50,
                          verbose = 1,
                          callbacks = cb,
                          validation_data = val,
                          validation_steps = 16)

h = his.history
h.keys()

plt.plot(h['accuracy'])
plt.plot(h['val_accuracy'], c = 'red')
plt.title('Accuracy vs Validation Accuracy')
plt.show()

plt.plot(h['loss'])
plt.plot(h['val_loss'], c = 'red')
plt.title('Loss vs Validation Loss')
plt.show()



