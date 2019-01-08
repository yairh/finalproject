import pandas as pd
from fetch_functions import *
from sklearn.linear_model import SGDClassifier
from batch_modelling import BModel
from sklearn.metrics import confusion_matrix, classification_report

REDUCE_SIZE = 4

def preprocess(df):
    """preprocess the dataframe before modelling"""

    rep_dic = {label: False if label == 'No Finding' else True for label in df['Finding Labels'].unique()}
    df['Finding Labels'] = df['Finding Labels'].replace(rep_dic)
    df = pd.concat([df, pd.get_dummies(df['Patient Gender'], prefix='Gender')], axis=1)
    cols_to_keep = ['Image Index', 'Follow-up #', 'Patient Age', 'Gender_F', 'Gender_M', 'Finding Labels']

    return df[cols_to_keep]


def main():
    """Main function for batch fit"""
    zipnames = ['/home/yair/Documents/ITC/Final/image/images_001.zip',
                '/home/yair/Documents/ITC/Final/image/images_002.zip',
                '/home/yair/Documents/ITC/Final/image/images_003.zip']
    with zp.ZipFile(zipnames[0]) as myzip:
        im_list1 = myzip.namelist()

    with zp.ZipFile(zipnames[1]) as myzip:
        im_list2 = myzip.namelist()

    df = pd.read_csv('/home/yair/Documents/ITC/Final/finalproject/data/Data_Entry_2017.csv')
    df_inp = preprocess(df)
    print('preprocess: Done')

    # define a pipeline from fetching the images, preprocess, add features, and partial fit the model
    # let's try with 1 zip
    idxs1 = list(chunks(im_list1, 300))
    idxs2 = list(chunks(im_list2, 300))

    # choose the model
    bmodel = BModel(SGDClassifier(loss='log', penalty='none', warm_start=True))

    # partial fit
    feature_cols = ['Follow-up #', 'Patient Age', 'Gender_F', 'Gender_M']

    bmodel.multi_gradual_fit(zipnames, df_inp, 4, 'Image Index', 'Finding Labels', feature_cols)
    bmodel.zip_validation(zipnames[2], df, 'Image Index', feature_cols, 'Finding Label', REDUCE_SIZE)

    #
    #
    # X_test, y_test = bmodel.split_train(im_list2[:500], df_inp, feature_cols, 'Image Index', 'Finding Labels',
    #                                     zipnames[1],
    #                                     4)
    # accuracy_test = bmodel.model.score(X_test, y_test)
    # y_pred = bmodel.predict(X_test)
    # print(f'mean accuracy on zip2: {accuracy_test}')
    # tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    # print(classification_report(y_test, y_pred))
    # print('TPR: ', tp / (tp + fn))
    # print('FPR: ', fp / (fp + tn))


if __name__ == "__main__":
    main()
