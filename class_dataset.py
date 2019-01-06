import os
import csv
import imageio
from skimage.measure import block_reduce
import numpy as np
import warnings
import pandas as pd


class ChestDataset:
    """Dataset class to manipulate data
    WARNING: it does return image (fetch method) ! """

    def __init__(self, data_dir, list_file, reduce_size=None):
        image_path = []
        labels = []
        ages = []
        followup = []
        gender = []
        self.reduce = reduce_size
        self.dir = data_dir

        if isinstance(list_file, pd.DataFrame):
            self.reader = list_file
            image_path = [os.path.join(self.dir, x) for x in self.reader['Image Index']]
            labels = list(self.reader['Finding Labels'])
            ages = list(self.reader['Patient Age'])
            followup = list(self.reader['Follow-up #'])
            gender = list(self.reader['Patient Gender'])
        elif isinstance(list_file, str):
            with open(list_file, 'r') as f:
                self.reader = csv.DictReader(f)
            for line in self.reader:
                image_path.append(os.path.join(self.dir, line['Image Index']))
                labels.append(line['Finding Labels'])
                ages.append(line['Patient Age'])
                followup.append(line['Follow-up #'])
                gender.append(line['Patient Gender'])
        else:
            raise TypeError('list file arg should be a path to a csv file or a Dataframe')

        self.image_path = image_path
        self.labels = labels
        self.gender = gender
        self.followup = followup
        self.ages = ages

    def __len__(self):
        return len(self.image_path)

    def __getitem__(self, index):
        image_path = self.image_path[index]
        label = self.labels[index]
        gender = self.gender[index]
        followup = self.followup[index]
        age = self.ages[index]

        return dict(image_path=image_path, label=label, gender=gender, followup=followup, ages=age)

    def __iter__(self):
        for index in range(len(self)):
            yield self[index]

    def filterby(self, filter):
        if not type(filter) == str:
            raise TypeError('filter arg should be a string')
        idx = [i for i, el in enumerate(self.labels) if el == filter]
        image_path = [self.image_path[i] for i in idx]
        labels = [self.labels[i] for i in idx]
        gender = [self.gender[i] for i in idx]
        followup = [self.followup[i] for i in idx]
        ages = [self.ages[i] for i in idx]

        return dict(image_path=image_path, label=labels, gender=gender,
                    followup=followup,
                    ages=ages)

    def filterby_obj(self, filter):
        if not isinstance(self.reader, pd.DataFrame):
            raise TypeError('Cannot use filterby_obj method if initialized with csv, initialize with dataframe instead')
        new_list = self.reader[self.reader['Finding Labels'] == filter]
        return ChestDataset(self.dir, new_list, self.reduce)

    def fetch(self, index):
        if self.reduce is None:
            image = imageio.imread(self.image_path[index])
        else:
            image = block_reduce(imageio.imread(self.image_path[index]), block_size=(self.reduce, self.reduce),
                                 func=np.mean)
        if len(image.shape) > 2:
            image = image[:, :, 0]

        return image

    def create_tree(self):
        for el in range(len(self)):
            if os.path.exists(self.image_path[el]):
                new_path = os.path.join(self.dir, self.labels[el])
                if not os.path.exists(new_path):
                    os.mkdir(new_path)
                os.rename(self.image_path[el], os.path.join(new_path, self.image_path[el].replace(self.dir, '')))
            else:
                warnings.warn('Image not found in folder')
                continue
