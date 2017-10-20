#!/usr/bin/python3

from __future__ import division, print_function, absolute_import

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sklearn.preprocessing as skpr
import math
import os
import sys
import scipy.misc
import h5py
import shutil
from tflearn.data_utils import build_hdf5_image_dataset


# Gramian Angular Field
def getGAF(row):
    # Scale values to [-1,1]
    new_x = skpr.maxabs_scale(row)
    # Transform to theta and r coordinates
    theta = np.arccos(new_x)
    r = 0 # not important
    # Convert to GAF image
    GAF = np.array([ np.cos(theta+tet) for tet in theta ])
    return GAF


# Markow Transition Field
def getMTF(row, Q=64):
    # Get quantiles from serie
    q = pd.qcut(list(set(row)), Q)
    # Create a dict of series and quantiles
    serie_quantil_dict = dict(zip(set(row), q.codes))

    # Create empty matrix size of quantiles
    MTF = np.zeros([Q,Q])

    # Get all values from dictionary
    labels = list(serie_quantil_dict.values())

    # Iterate on all labels,
    for i in range(len(labels)-1):
        MTF [labels[i]] [labels[i+1]] += 1

    # Scale from 0 to 1
    for i in range(Q):
        if sum(MTF[i][:]) == 0:
            continue
        MTF[i][:] = MTF[i][:]/sum(MTF[i][:])

    return np.array(MTF)


def timeSeriesToImages(dataframe, split, method='GAF'):

    #Remove previouse images, if any
    if os.path.isdir('images/%s' % split):
        shutil.rmtree('images/%s' % split)
        os.mkdir('images/%s' % split)
    else:
        print('creating')
        os.mkdir('images/%s' % split)

    for i in range(len(dataframe.index)):
        target, *serie = dataframe.iloc[i]
        if method == 'GAF':
            image = getGAF(serie)
        else:
            image = getMTF(serie)

        if not os.path.isdir('images/%s/%s' % (split, int(target))):
            os.mkdir('images/%s/%s' % (split, int(target)))

        scipy.misc.toimage(image, cmin=-1, cmax=1).save('images/%s/%s/%s.jpg' % (split, int(target), i))

    output_path = '%s.h5' % split

    #df_height, df_width = dataframe.shape
    #num_features = df_width - 1

    # Delete last h5 dataset
    if os.path.isfile('%s.h5' % split):
        os.remove('%s.h5' % split)  

    build_hdf5_image_dataset('images/%s' % split, image_shape=(224, 224),
                             mode='folder', output_path=output_path, categorical_labels=True)



if __name__ == '__main__':
    train_path = sys.argv[1]
    test_path = sys.argv[2]

    # Read time series datasets
    train_df = pd.read_csv(train_path, header=None)
    test_df = pd.read_csv(test_path, header=None)

    timeSeriesToImages(train_df, 'train')
    timeSeriesToImages(test_df, 'test')


