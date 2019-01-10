import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import shutil
from sklearn.datasets import load_files
import shutil,os,glob

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


df_data_entry = pd.read_csv('../../data/Data_Entry_2017.csv')
df = df_data_entry.iloc[:,:11]

# rows having no. of disease
df['labels_count'] = df['Finding Labels'].apply(lambda text: len(text.split('|')) if(text != 'No Finding') else 0)

#Open train/test lists
with open('../../data/train_val_list.txt', 'r') as train_items:
    train_list = train_items.readlines()
train_list = [item.strip() for item in train_list]

with open('../../data/test_list.txt', 'r') as test_items:
    test_list = test_items.readlines()
test_list = [item.strip() for item in test_list]


# def binary_labels(label):
#     return folder if folder == 'No Finding' else 'Finding'

def labels_split(folder, train_test):
    dest = "../../imgs/images/" + train_test + "/"
    if not os.path.exists(os.path.join(dest, folder)):
        os.mkdir(os.path.join(dest, folder))
        shutil.move(item, os.path.join(dest, folder))
    else:
        shutil.move(item, os.path.join(dest, folder))


def train_test_split(train_test):
    dest = '../../imgs/images/'
    if not os.path.exists(os.path.join(dest, train_test)):
        os.mkdir(os.path.join(dest, train_test))

    else:
        pass


# Split all classes
dest = '../../imgs/images/'

for item in glob.glob('../../imgs/images/*'):
    photo = item[18:]
    count_labels = df.loc[df['Image Index'] == photo, 'labels_count'].values[0]

    if count_labels == 1:  # keep only 1 disease
        folder = df.loc[df['Image Index'] == photo, 'Finding Labels'].values[0]  # Disease

        if photo in train_list:
            train_test_split('train')
            labels_split(folder, 'train')

        elif photo in test_list:
            train_test_split('test')
            labels_split(folder, 'test')

        else:
            pass

    else:
        pass