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


import tflearn
import tensorflow as tf
from tflearn.data_utils import shuffle, to_categorical
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.recurrent import lstm
from tflearn.layers.estimator import regression
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_augmentation import ImageAugmentation
from tflearn.data_utils import build_hdf5_image_dataset



def getAlexnet(num_classes, num_features):

	network = input_data(shape=[None, num_features, num_features, 3])
	network = conv_2d(network, 64, 11, strides=4, activation='relu')
	network = max_pool_2d(network, 3, strides=2)
	network = local_response_normalization(network)
	network = conv_2d(network, 128, 5, activation='relu')
	network = max_pool_2d(network, 3, strides=2)
	network = local_response_normalization(network)
	network = conv_2d(network, 256, 3, activation='relu')
	network = conv_2d(network, 256, 3, activation='relu')
	network = conv_2d(network, 128, 3, activation='relu')
	network = max_pool_2d(network, 3, strides=2)
	network = local_response_normalization(network)
	network = fully_connected(network, 4096, activation='tanh')
	network = dropout(network, 0.5)
	network = fully_connected(network, 4096, activation='tanh')
	network = dropout(network, 0.5)
	network = fully_connected(network, num_classes, activation='softmax')
	network = regression(network, optimizer='adam',
	                     loss='categorical_crossentropy',
	                     learning_rate=0.001)
	return network


def getVGG_finetune(num_classes, num_features):
	pass


def getLSTM(num_classes, num_features):
	network = input_data([None, 1, num_features])
	network = lstm(network, 256, dropout=0.8)
	network = fully_connected(network, num_classes, activation='softmax')
	network = regression(network, optimizer='adam', learning_rate=0.01, loss='categorical_crossentropy')
	return network


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
def getMTF(row):
    pass


def countClasses(dataframe):
	word_dict = {}
	for data in dataframe[0]:
		x = word_dict.get(data, 0)
		word_dict[data] = x + 1

	return len(word_dict)


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

	df_height, df_width = dataframe.shape
	num_features = df_width - 1

	# Delete last h5 dataset
	if os.path.isfile('%s.h5' % split):
		os.remove('%s.h5' % split)	

	build_hdf5_image_dataset('images/%s' % split, image_shape=(num_features, num_features),
							 mode='folder', output_path=output_path, categorical_labels=True)



def train_cnn(train_df, test_df, num_classes):
	# Convert to images
	timeSeriesToImages(train_df, 'train')
	timeSeriesToImages(test_df, 'test')

	# Read HDF5 datasets for train and test
	h5f_train = h5py.File('train.h5', 'r')
	h5f_test = h5py.File('test.h5', 'r')

	train_x, train_y = h5f_train['X'], h5f_train['Y']
	test_x, test_y = h5f_test['X'], h5f_test['Y']

	df_height, df_width = train_df.shape
	num_features = df_width - 1

	# Create model on Alexnet network
	model_cnn = tflearn.DNN(getAlexnet(num_classes, num_features), tensorboard_verbose=0)

	# Train model on Alexnet network
	model_cnn.fit(train_x, train_y, validation_set=(test_x, test_y), n_epoch=200, shuffle=True,
          show_metric=True, batch_size=96)


def train_lstm(train_df, test_df, num_classes):

	df_height, df_width = train_df.shape
	num_features = df_width - 1

	# Reshape train dataframe
	train_x = train_df.drop(train_df.columns[[0]], axis=1).values.tolist()
	train_x = np.reshape(train_x, (-1, 1, num_features))
	train_y = to_categorical([y-1 for y in train_df[0].values.tolist()], num_classes)

	# Reshape test dataframe
	test_x = test_df.drop(test_df.columns[[0]], axis=1).values.tolist()
	test_x = np.reshape(test_x, (-1, 1, num_features))
	test_y = to_categorical([y-1 for y in test_df[0].values.tolist()], num_classes)

	model_lstm = tflearn.DNN(getLSTM(num_classes, num_features), tensorboard_verbose=0)

	model_lstm.fit(train_x, train_y, n_epoch=200, validation_set=(test_x, test_y),
		show_metric=True)




if __name__ == '__main__':
	
	train_path = sys.argv[1]
	test_path = sys.argv[2]

	# Read time series datasets
	train_df = pd.read_csv(train_path, header=None)
	test_df = pd.read_csv(test_path, header=None)

	num_classes = countClasses(train_df)
	num_classes_test = countClasses(test_df)

	assert(num_classes == num_classes_test)

	train_cnn(train_df, test_df, num_classes)
	# Weird error without this
	tf.reset_default_graph()

	train_lstm(train_df, test_df, num_classes)












		