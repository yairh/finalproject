import numpy as np
import imageio
import zipfile as zp
from tqdm import tqdm
from skimage.measure import block_reduce


def chunks(l, n):
    # For item i in a range that is a length of l,
    for i in range(0, len(l), n):
        # Create an index range for l of n items:
        yield l[i:i + n]


def ravel_mat(mat):
    """ravel images of entire matrix and returned the raveled matrix"""
    res = np.zeros((mat.shape[0], mat.shape[1] ** 2))
    for i in range(mat.shape[0]):
        res[i, :] = mat[i, :, :].ravel()
    return res


def fetch_im(myzip, imagename):
    # fetch specific image from zip file
    image = imageio.imread(myzip.read(imagename))
    if len(image.shape) > 2:
        image = image[:, :, 0]
    return image


def fetch_all(zipname, reduce_size, nb_image=None):
    # fetch all or a selected number of image from zipfile, reduce the image by reduce size factor
    with zp.ZipFile(zipname) as myzip:
        im_list = myzip.namelist()

    if not nb_image:
        nb_image = len(im_list)
    new_size = round(1024 / reduce_size)

    im_mat = np.zeros((nb_image, new_size, new_size))

    with zp.ZipFile(zipname) as myzip:
        for i, im in enumerate(tqdm(im_list[:nb_image])):
            im_mat[i, :, :] = block_reduce(
                fetch_im(myzip, im), block_size=(reduce_size, reduce_size), func=np.mean)
    return im_mat


def fetch_list(zipname, reduce_size, im_list):
    """fetch a selected list of images from a zipfile"""
    new_size = round(1024 / reduce_size)
    im_mat = np.zeros((len(im_list), new_size, new_size))

    with zp.ZipFile(zipname) as myzip:
        for i, im in enumerate(tqdm(im_list)):
            im_mat[i, :, :] = block_reduce(
                fetch_im(myzip, im), block_size=(reduce_size, reduce_size), func=np.mean)
    return im_mat


def index_list(df, l, col):
    """return list of location of a subset
     WARNING: custom for dataframe of images NCIH, modify the function to your own needs"""
    return [df.set_index(col).index.get_loc(x.replace('images/', '')) for x in l]
