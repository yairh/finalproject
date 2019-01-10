import os
import csv
import imageio
from skimage.measure import block_reduce
import numpy as np
import warnings
import pandas as pd
import glob


class ChestDataset:
    """Dataset class to manipulate data
    WARNING: it does return image (fetch method) ! """

    def __init__(self, data_dir, list_file, reduce_size=None):
        image_path = []
        labels = []
        ages = []
        followup = []
        gender = []
        exists = []
        self.reduce = reduce_size
        self.dir = data_dir

        if isinstance(list_file, pd.DataFrame):
            self.reader = list_file
            image_path = [os.path.join(self.dir, x) for x in self.reader['Image Index']]
            labels = list(self.reader['Finding Labels'])
            ages = list(self.reader['Patient Age'])
            followup = list(self.reader['Follow-up #'])
            gender = list(self.reader['Patient Gender'])
            exists = [True if os.path.exists(path) else False for path in image_path]
        elif isinstance(list_file, str):
            with open(list_file, 'r') as f:
                self.reader = csv.DictReader(f)
            for line in self.reader:
                image_path.append(os.path.join(self.dir, line['Image Index']))
                labels.append(line['Finding Labels'])
                ages.append(line['Patient Age'])
                followup.append(line['Follow-up #'])
                gender.append(line['Patient Gender'])
                exists.append(True if os.path.exists(image_path[-1]) else False)
        else:
            raise TypeError('list file arg should be a path to a csv file or a Dataframe')

        self.image_path = image_path
        self.labels = labels
        self.gender = gender
        self.followup = followup
        self.ages = ages
        self.exists = exists
        self.reader['exists'] = self.exists

    def __len__(self):
        return len(self.image_path)

    def __getitem__(self, index):
        image_path = self.image_path[index]
        label = self.labels[index]
        gender = self.gender[index]
        followup = self.followup[index]
        age = self.ages[index]
        exists = self.exists[index]

        return dict(image_path=image_path, label=label, gender=gender, followup=followup, ages=age, exists=exists)

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
        exists = [self.exists[i] for i in idx]

        return dict(image_path=image_path, label=labels, gender=gender,
                    followup=followup,
                    ages=ages, exists=exists)

    def filterby_obj(self, filter, attribute):
        if not isinstance(self.reader, pd.DataFrame):
            raise TypeError('Cannot use filterby_obj method if initialized with csv, initialize with dataframe instead')
        new_list = self.reader[self.reader[attribute] == filter]
        return ChestDataset(self.dir, new_list, self.reduce)

    def fetch(self, index):
        if not self.exists[index]:
            raise FileNotFoundError(
                'Image not found in physical folder, if you added it, please reinitialize Dataset object')
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
            if self.exists[el]:
                new_path = os.path.join(self.dir, self.labels[el])
                if not os.path.exists(new_path):
                    os.mkdir(new_path, 0o777)
                os.rename(self.image_path[el], os.path.join(new_path, self.image_path[el].replace(self.dir, '')))
            else:
                warnings.warn('Image not found in folder')
                continue

    def train_test(self, train_list, test_list):
        df_train = self.reader[self.reader['Image Index'].isin(train_list)]
        df_test = self.reader[self.reader['Image Index'].isin(test_list)]

        train_dataset = ChestDataset(self.dir, df_train, self.reduce)
        test_dataset = ChestDataset(self.dir, df_test, self.reduce)

        train_path = os.path.join(self.dir, 'train/')
        test_path = os.path.join(self.dir, 'test/')
        os.mkdir(train_path, 0o777)
        os.mkdir(test_path, 0o777)

        self.path_translate(train_dataset, train_path)
        self.path_translate(test_dataset, test_path)

        train_dataset = ChestDataset(train_path, df_train, self.reduce)
        test_dataset = ChestDataset(test_path, df_test, self.reduce)
        return train_dataset, test_dataset

    def path_translate(self, dataset, trans_path):
        for img in dataset:
            if not img['exists']:
                warnings.warn('file not found')
                continue
            path = img['image_path']
            new_path = os.path.join(trans_path, path.replace(self.dir, ''))
            os.rename(path, new_path)

    def reset_folder(self):
        for folder in glob.glob(os.path.join(self.dir, '**/**/')):
            for files in glob.glob(os.path.join(folder, '*.png')):
                file = files.replace(folder, '')
                new_path = os.path.join(self.dir, file)
                os.rename(files, new_path)
            os.rmdir(folder)
        if os.path.exists(os.path.join(self.dir,'train/')):
            os.rmdir(os.path.join(self.dir,'train/'))
        if os.path.exists(os.path.join(self.dir,'test/')):
            os.rmdir(os.path.join(self.dir,'test/')) 
