# -*- coding: utf-8 -*-

from loadDataset import load_data, load_multi
import theano
import theano.tensor as T

batch_size = 600
dataset='mnist.pkl.gz'
datasets, train_set, valid_set, test_set = load_data(dataset)


train_set_x, train_set_y = datasets[0]
valid_set_x, valid_set_y = datasets[1]
test_set_x, test_set_y = datasets[2]

n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] / batch_size
n_test_batches = test_set_x.get_value(borrow=True).shape[0] / batch_size

import scipy.io
import numpy as np

l_file = scipy.io.loadmat('multi_data/test_samples.mat')
test_samples = l_file['test_samples']
l_file = scipy.io.loadmat('multi_data/test_labels.mat')
test_labels = l_file['test_labels']

l_file = scipy.io.loadmat('multi_data/valid_samples.mat')
valid_samples = l_file['valid_samples']
l_file = scipy.io.loadmat('multi_data/valid_labels.mat')
valid_labels = l_file['valid_labels']

l_file = scipy.io.loadmat('multi_data/train_samples.mat')
train_samples = l_file['train_samples']
l_file = scipy.io.loadmat('multi_data/train_labels.mat')
train_labels = l_file['train_labels']


train_set = [train_samples, np.reshape(train_labels, (145180,))]
valid_set = [valid_samples, np.reshape(valid_labels, (31111,))]
test_set = [test_samples, np.reshape(test_labels, (31110,))]

data_x, data_y = valid_set
shared_x = theano.shared(np.asarray(data_x, dtype=theano.config.floatX), borrow=False)
shared_y = theano.shared(np.asarray(data_y, dtype=theano.config.floatX), borrow=True)
T.cast(shared_y, 'int32')



datasets = load_multi(train_set, valid_set, test_set)


