from fetch_functions import *
import logging
from sklearn.metrics import classification_report

logger = logging.getLogger('stream')
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())


class BModel:

    def __init__(self, model):
        """initiation: receive model implementing partial fit"""
        if model.partial_fit:
            self.model = model
        else:
            raise Exception('Model should implement partial_fit')

    def gradual_fit(self, chunks, df, reduce_size, index_col, label_col, feature_cols, zipname, first=True):
        """gradual fit model to all chunks given, the df passed is used as reference for images
        last chunk is keeped for test set (to improve)"""

        y, X_test, y_test = self.split_test(chunks, df, zipname, index_col, label_col,reduce_size)

        for i, chunk in enumerate(chunks[:-1]):
            logger.info(f'Chunk {i}: building matrix...')
            X_train, y_train = self.split_train(chunk, df, feature_cols, index_col, label_col, zipname, reduce_size)

            if i == 0 and first:
                self.model.fit(X_train, y_train)
                accuracy_train = self.model.score(X_train, y_train)
                accuracy_test = self.model.score(X_test, y_test)
                logger.info(f'First fit {i} ---- Train mean accuracy ---- {accuracy_train} ')
                logger.info(f'First fit {i} ---- Test mean accuracy ---- {accuracy_test} ')
            else:
                logger.info(f'Chunk {i}: partial fit...')
                self.model.partial_fit(X_train, y_train, classes=y_train.unique())
                accuracy_train = self.model.score(X_train, y_train)
                accuracy_test = self.model.score(X_test, y_test)
                logger.info(f'Partial fit {i} ---- Train mean accuracy ---- {accuracy_train} ')
                logger.info(f'Partial fit {i} ---- Test mean accuracy ---- {accuracy_test} ')

            del X_train, y_train  # free memory again

    def predict(self, x):
        return self.model.predict(x)

    def fit(self, x_train, y_train):
        self.model.fit(x_train, y_train)

    @staticmethod
    def split_test(chunks, df, zipname, index_col, label_col, reduce_size):
        """split test set from last chunk"""
        y = df[label_col]
        logger.info('Fetching Test Matrix...')
        locs = index_list(df, chunks[-1], index_col)
        mat_test = ravel_mat(fetch_list(zipname, reduce_size, chunks[-1]))
        X_test = np.c_[mat_test, df.drop([index_col, label_col], axis=1).iloc[locs]]
        y_test = y.iloc[locs]
        del mat_test
        return y, X_test, y_test

    @staticmethod
    def split_train(chunk, df, feature_cols, index_col, label_col, zipname, reduce_size):
        y = df[label_col]
        locs = index_list(df, chunk, index_col)
        im_mat = ravel_mat(fetch_list(zipname, reduce_size, list(chunk)))
        X_train = np.c_[im_mat, df[feature_cols].iloc[locs]]
        del im_mat  # free memory
        y_train = y.iloc[locs]
        del y
        return X_train, y_train

    def multi_gradual_fit(self, zipnames, df, reduce_size, index_col, label_col, feature_cols):

        for i, zipname in enumerate(zipnames):

            if i == 0:
                first = True
            else:
                first = False

            with zp.ZipFile(zipname) as myzip:
                im_list = myzip.namelist()

            chunk_list = list(chunks(im_list, 300))

            self.gradual_fit(chunk_list, df, reduce_size, index_col, label_col, feature_cols, zipname, first)

    def zip_validation(self, zipname, df, index_col, feature_col, label_col, reduce_size):

        with zp.ZipFile(zipname) as myzip:
            im_list = myzip.namelist()
        test = list(chunks(im_list, 300))
        pred = np.zeros(len(im_list))
        y_true = np.zeros(len(im_list))
        for chunk in test:
            X_train, y_train = self.split_train(chunk, df, feature_col, index_col, label_col, zipname, reduce_size)
            pred = np.c_[pred,self.predict(X_train)]
            y_true = np.c_[y_true,y_train]
        print(classification_report(y_true,pred))

