import pandas as pd
import logging as log
import glob
import random
import numpy as np
import os.path as path
from scipy import misc


# this function is called whenever the 'ergo train <project> --dataset folder'
# command is executed, the first argument is the dataset and it must return
# a pandas.DataFrame object.
def prepare_dataset(folder):
    log.info("loading images from %s ...", folder)

    # read all images paths into lists
    # discard 90% of non-keys at random (using the whole data can freeze your computer)
    all_paths = glob.glob(path.join(folder, '*/*.jpg')) # images are in subfolders of the input data folder
    keys_paths = glob.glob(path.join(folder, '*/*k*.jpg')) # filename with 'k' means the image contains a key
    non_keys_paths = [p for p in all_paths if p not in keys_paths] 
    discarded = random.sample(range(len(non_keys_paths)), int(0.85 * len(non_keys_paths)))
    for index in sorted(discarded, reverse=True):
       del non_keys_paths[index]
    paths = list(set().union(keys_paths,non_keys_paths))
       
    # read all chosen images into an array
    images = [misc.imread(path) for path in paths]
    images = np.asarray(images)
    n_images = images.shape[0]

    # normalize pixels to [0.0, 1.0]
    images = images / 255
        
    log.info("vectorializing %d samples ...", n_images)

    # get label from filename
    labels = np.zeros(n_images)
    for i in range(n_images):
        filename = path.basename(paths[i])
        if 'k' in filename: # filename with 'k' means the image contains a key
            labels[i] = 1

    # create the flattened training matrix
    dataset = []
    for i in range(n_images):
        X = images[i].flatten()
        y = labels[i]
        dataset.append(np.insert(X, 0, y, axis=0)) # label first

    return pd.DataFrame(dataset)

# called during 'ergo serve <project>' for each 'x' input parameter, use this 
# function to convert, for instance, a file name in a vector of scalars for 
# your model input layer.
def prepare_input(path):
    log.debug("vectorializing %s ...", path)
    x = misc.imread(path)
    #log.debug("shape   : %s", x.shape)
    #x = misc.imresize(x, (40, 40))
    #log.debug("resized : %s", x.shape)
    x = x / 255
    x = x.flatten()
    log.debug("flat    : %s", x.shape)
    return np.asarray([x])
