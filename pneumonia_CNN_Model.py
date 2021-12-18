from keras.preprocessing import image
from keras.layers import Dense, Conv2D, MaxPool2D, Dropout, Flatten, BatchNormalization
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras

# os.listdir("./data")

# len(os.listdir("./data/train/PNEUMONIA"))

train_dir = "./data/train"
test_dir = "./data/test"
val_dir = "./data/val"

# Data Visualization
num_pneumonia = len(os.listdir(os.path.join(train_dir, 'PNEUMONIA')))
num_normal = len(os.listdir(os.path.join(train_dir, 'NORMAL')))

# Image Preprocessing
image_generator = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    samplewise_center=True,
    samplewise_std_normalization=True
)


train = image_generator.flow_from_directory(
    train_dir, batch_size=8, shuffle=True, class_mode='binary', target_size=(180, 180))

print(train.next())

validation = image_generator.flow_from_directory(
    val_dir, batch_size=1, shuffle=False, class_mode='binary', target_size=(180, 180))

test = image_generator.flow_from_directory(
    test_dir, batch_size=1, shuffle=False, class_mode='binary', target_size=(180, 180))


# train = image_generator.flow_from_directory(
#     train_dir, batch_size=8, shuffle=True, color_mode='grayscale', class_mode='binary', target_size=(256, 256))

# validation = image_generator.flow_from_directory(
#     val_dir, batch_size=1, shuffle=False, color_mode='grayscale', class_mode='binary', target_size=(256, 256))

# test = image_generator.flow_from_directory(
#     test_dir, batch_size=1, shuffle=False, color_mode='grayscale', class_mode='binary', target_size=(256, 256))


# Class weights

weight_for_0 = num_pneumonia / (num_normal + num_pneumonia)
weight_for_1 = num_normal / (num_normal + num_pneumonia)

class_weight = {0: weight_for_0, 1: weight_for_1}

# Build CNN Model

model = Sequential()

model.add(Conv2D(filters=32, kernel_size=(3, 3),
          input_shape=(180, 180, 3), activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(filters=32, kernel_size=(3, 3),
          input_shape=(180, 180, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2, 2)))

model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2, 2)))

model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam', metrics=['accuracy'])


model.summary()

# model = Sequential()

# model.add(Conv2D(filters=32, kernel_size=(3, 3),
#           input_shape=(256, 256, 1), activation='relu'))
# model.add(BatchNormalization())
# model.add(Conv2D(filters=32, kernel_size=(3, 3),
#           input_shape=(256, 256, 1), activation='relu'))
# model.add(BatchNormalization())
# model.add(MaxPool2D(pool_size=(2, 2)))

# model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
# model.add(BatchNormalization())
# model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
# model.add(BatchNormalization())
# model.add(MaxPool2D(pool_size=(2, 2)))

# model.add(Flatten())
# model.add(Dense(64, activation='relu'))
# model.add(Dropout(0.2))

# model.add(Dense(1, activation='sigmoid'))

# model.compile(loss='binary_crossentropy',
#               optimizer='adam',
#               metrics=['accuracy'])

# model.summary()

# model.fit(
#     train,
#     epochs=3,
#     validation_data=validation,
#     class_weight=class_weight,
#     steps_per_epoch=100,
#     validation_steps=25,
#     # train,
#     # epochs=10,
#     # validation_data=validation,
#     # class_weight=class_weight,
#     # steps_per_epoch=len(train),
#     # validation_steps=len(validation),
# )

# evaluation=model.evaluate(test)
# print(f"Test Accuracy: {evaluation[1] * 100:.2f}%")

# model.save("pneumonia_CNN.h5")
