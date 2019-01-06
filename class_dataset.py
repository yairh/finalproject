import os
import csv
import imageio
from skimage.measure import block_reduce
import numpy as np


class ChestDataset:
    """Dataset class to manipulate data
    WARNING: it does not return images ! """

    def __init__(self, data_dir, list_file, reduce_size=None):
        image_names = []
        labels = []
        ages = []
        followup = []
        gender = []
        self.reduce = reduce_size

        with open(list_file, 'r') as f:
            self.reader = csv.DictReader(f)
            for line in self.reader:
                image_names.append(os.path.join(data_dir, line['Image Index']))
                labels.append(line['Finding Labels'])
                ages.append(line['Patient Age'])
                followup.append(line['Follow-up #'])
                gender.append(line['Patient Gender'])

        self.image_names = image_names
        self.labels = labels
        self.gender = gender
        self.followup = followup
        self.ages = ages

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, index):
        image_name = self.image_names[index]
        label = self.labels[index]
        gender = self.gender[index]
        followup = self.followup[index]
        age = self.ages[index]

        return dict(image_name=image_name, label=label, gender=gender, followup=followup, ages=age)

    def __iter__(self):
        for index in range(len(self)):
            yield self[index]

    def filterby(self, filter):
        if not type(filter) == str:
            raise TypeError('filter arg should be a string')
        idx = [i for i, el in enumerate(self.labels) if el == filter]
        image_names = [self.image_names[i] for i in idx]
        labels = [self.labels[i] for i in idx]
        gender = [self.gender[i] for i in idx]
        followup = [self.followup[i] for i in idx]
        ages = [self.ages[i] for i in idx]

        return dict(image_name=image_names, label=labels, gender=gender,
                    followup=followup,
                    ages=ages)

    def fetch(self, index):
        if self.reduce is None:
            image = imageio.imread(self.image_names[index])
        else:
            image = block_reduce(imageio.imread(self.image_names[index]), block_size=(self.reduce, self.reduce),
                                 func=np.mean)
        if len(image.shape) > 2:
            image = image[:, :, 0]

        return image
