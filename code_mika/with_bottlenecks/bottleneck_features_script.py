from sklearn.datasets import load_files
from keras.utils import np_utils
import numpy as np
import pandas as pd
from glob import glob
from IPython.core.display import Image, display

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

from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers import Dropout, Flatten, Dense
from keras.models import Sequential


img_width, img_height, channels = 224, 224, 3

top_model_weights_path = 'bottleneck_fc_model.h5'
train_data_dir = '../imgs/images/train'
test_data_dir = '../imgs/images/test'

nb_train_samples = 4032
nb_test_samples = 967
epochs = 2

train_batch_size = 30
test_batch_size = 10

### Load VGG model
model = ResNet50(weights= 'imagenet', include_top=False, input_shape=(img_height, img_width, channels))

### Freeze some layers
for layer in model.layers:
    layer.trainable = False

### Check the trainable status of the individual layers
for layer in model.layers:
    print(layer, layer.trainable)

print('**********************TRAIN GENERATOR**********************')
### Train Generator
train_datagen = ImageDataGenerator(rescale=1. / 255)

"""
                                   rotation_range=20,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True
"""

train_generator = train_datagen.flow_from_directory(train_data_dir,
                                                    target_size=(img_width, img_height),
                                                    batch_size = train_batch_size,
                                                    class_mode = 'categorical',
                                                    shuffle=False)
print('End Train Generator')
# Bottlenecks are the last activation maps before the fully-connected layers in the original model
for _ in tqdm(range(1)):
    bottleneck_features_train = model.predict_generator(train_generator, nb_train_samples // train_batch_size)
print('Ready to save train BF')
np.save(open('bottleneck_features_train.npy', 'wb'), bottleneck_features_train)
print('Train BF Saved !')

print('**********************TEST GENERATOR**********************')
### Test Generator
test_datagen = ImageDataGenerator(rescale=1. / 255)
test_generator = test_datagen.flow_from_directory(test_data_dir,
                                                    target_size=(img_width, img_height),
                                                    batch_size=test_batch_size,
                                                    class_mode='categorical',
                                                    shuffle=False)
print('End Test Generator')
bottleneck_features_test = model.predict_generator(test_generator, nb_test_samples // test_batch_size)
print('Ready to save test BF')
np.save(open('bottleneck_features_test.npy', 'wb'), bottleneck_features_test)
print('Test BF Saved !')