import numpy as np
import pandas as pd
from glob import glob
from sklearn.datasets import load_files

from keras.preprocessing.image import ImageDataGenerator
from keras.applications import VGG16, DenseNet121
from keras.utils import np_utils
from keras.applications.inception_v3 import InceptionV3

from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers import Dropout, Flatten, Dense, Activation
from keras.models import Sequential

# define function to load train, test, and validation datasets
def load_dataset(path, n_classes):
    """Returns the path and the Label from the folder"""
    data = load_files(path)
    chest_files = np.array(data['filenames'])
    chest_targets = np_utils.to_categorical(np.array(data['target']), n_classes)
    return chest_files, chest_targets

# load list of dog names
labels = [item[18:-1] for item in sorted(glob("../imgs/all/train/*/"))]
n_classes = len(labels)

# load train, test, and validation datasets
train_files, train_targets = load_dataset('../imgs/all/train', n_classes)
test_files, test_targets = load_dataset('../imgs/all/test', n_classes)

# Img size
img_width, img_height, channels = 224, 224, 3

# Model
model = DenseNet121(weights= 'imagenet', include_top=False, input_shape=(img_height, img_width, channels))

# build a classifier model to put on top of the convolutional model
top_model = Sequential()
top_model.add(Flatten(input_shape=model.output_shape[1:]))
top_model.add(Dense(256, activation='relu'))
top_model.add(Dropout(0.5))
top_model.add(Dense(1, activation='sigmoid'))

# add the model on top of the convolutional base
model.add(top_model)