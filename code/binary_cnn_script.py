from sklearn.datasets import load_files
from keras.utils import np_utils
import numpy as np
import pandas as pd
from glob import glob
from IPython.core.display import Image, display
import matplotlib.pyplot as plt

from keras import models, layers, optimizers
from keras.preprocessing import image
from keras.applications.vgg19 import VGG19
from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import preprocess_input, decode_predictions
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import VGG16
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.callbacks import ModelCheckpoint
from tqdm import tqdm
import math
import pickle
import os
from PIL import ImageFile
import time

from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers import Dropout, Flatten, Dense, Activation
from keras.models import Sequential
import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard

# from keras.callbacks import TensorBoard

"""
tensorboard --logdir=logs/ --host localhost --port 8088
"""

my_path = ''

# Add GPU options
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
NAME = 'Binary_chest_cnn_64x3_{}'.format(int(time.time()))
tensorboard = TensorBoard(log_dir='logs/{}'.format(NAME))

def to_net(path_to_img):
    train_path = os.path.join(path_to_img, 'train')
    test_path = os.path.join(path_to_img, 'test')

    # define function to load train, test, and validation datasets
    def load_dataset(path):
        """Returns the path and the Label from the folder"""
        data = load_files(path)
        chest_files = np.array(data['filenames'])
        chest_targets = np_utils.to_categorical(np.array(data['target']), 2)
        return chest_files, chest_targets

    # load train, test, and validation datasets
    train_files, train_targets = load_dataset(train_path)
    test_files, test_targets = load_dataset(test_path)

    # load list of image labels
    labels = [item[21:-1] for item in sorted(glob(train_path + '/*/'))]

    def path_to_tensor(img_path):
        # loads RGB image as PIL.Image.Image type
        img = image.load_img(img_path, target_size=(224, 224))
        # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
        x = image.img_to_array(img)
        # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
        return np.expand_dims(x, axis=0)

    def paths_to_tensor(img_paths):
        list_of_tensors = [path_to_tensor(img_path) for img_path in tqdm(img_paths)]
        return np.vstack(list_of_tensors)

    ImageFile.LOAD_TRUNCATED_IMAGES = True
    # pre-process the data for Keras
    train_tensors = paths_to_tensor(train_files).astype('float32') / 255
    test_tensors = paths_to_tensor(test_files).astype('float32') / 255

    return train_tensors, train_targets, test_tensors, test_targets, labels


train_tensors, train_targets, test_tensors, test_targets, labels = to_net(my_path)

model = Sequential()
model.add(Conv2D(64, (3, 3), input_shape=train_tensors.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())

# ---------------------------------------
model.add(layers.Dense(64))
model.add(layers.Dense(2))
model.add(layers.Activation('softmax'))
# ----------------------------------------

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(train_tensors, train_targets, validation_split=0.3,
          epochs=2, batch_size=32, verbose=1, shuffle=True, callbacks=[tensorboard])
