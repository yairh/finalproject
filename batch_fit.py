import pandas as pd
from fetch_functions import *
from sklearn.linear_model import SGDClassifier
from batch_modelling import BModel


def preprocess(df):
    """preprocess the dataframe before modelling"""

    rep_dic = {label: False if label == 'No Finding' else True for label in df['Finding Labels'].unique()}
    df['Finding Labels'] = df['Finding Labels'].replace(rep_dic)
    df = pd.concat([df, pd.get_dummies(df['Patient Gender'], prefix='Gender')], axis=1)
    cols_to_keep = ['Image Index', 'Follow-up #', 'Patient Age', 'Gender_F', 'Gender_M', 'Finding Labels']

    return df[cols_to_keep]


def main():
    """Main function for batch fit"""
    zipname1 = '/home/yair/Documents/ITC/Final/image/images_001.zip'
    zipname2 = '/home/yair/Documents/ITC/Final/image/images_002.zip'
    with zp.ZipFile(zipname1) as myzip:
        im_list1 = myzip.namelist()

    df = pd.read_csv('/home/yair/Documents/ITC/Final/finalproject/data/Data_Entry_2017.csv')
    df_inp = preprocess(df)
    print('preprocess: Done')

    # define a pipeline from fetching the images, preprocess, add features, and partial fit the model
    # let's try with 1 zip
    idxs = list(chunks(im_list1, 300))

    # choose the model
    bmodel = BModel(SGDClassifier(loss='log', penalty='none', warm_start=True))

    # partial fit
    feature_cols = ['Follow-up #', 'Patient Age', 'Gender_F', 'Gender_M']
    bmodel.gradual_fit(idxs, df_inp, 4, 'Image Index', 'Finding Labels', feature_cols, zipname1)


    with zp.ZipFile(zipname2) as myzip:
        im_list2 = myzip.namelist()

    images_test = fetch_all(zipname2, 4, 500)
    X_test = bmodel.split_train(im_list2[:500],df,feature_cols,'Image Index','Finding Labels',zipname2,4)


if __name__ == "__main__":
    main()
