# -*- coding: utf-8 -*-

import numpy as np
import scipy.io as io
import time
import os 
import sys
import theano
import theano.tensor as T
from loadDataset import load_multi



datasets = load_multi()
valid_set_x, valid_set_y = datasets[1]
test_set_x, test_set_y = datasets[2]

x_test, y_test = test_set_x.eval(), test_set_y.eval()
x_valid, y_valid = valid_set_x.eval(), valid_set_y.eval()

x_image = io.loadmat('multi_data/hc_serial.mat')
x_image = x_image['hc_serial']

weights = io.loadmat('multi_data/TrainedWeights.mat')
W1, W2, W3, W4 = weights['W1'], weights['W2'], weights['W3'], weights['W4']
b1, b2, b3, b4 = weights['b1'], weights['b2'], weights['b3'], weights['b4']

del datasets, valid_set_x, valid_set_y, test_set_x, test_set_y, weights

x = x_image/8000.

A1 = theano.shared(value=np.dot(x, W1) + b1, name='A1')
Z1 = T.nnet.sigmoid(A1)
A2 = theano.shared(value=np.dot(Z1.eval(), W2) + b2, name='A2')
Z2 = T.nnet.sigmoid(A2)
A3 = theano.shared(value=np.dot(Z2.eval(), W3) + b3, name='A3')
Z3 = T.nnet.sigmoid(A3)
A4 = theano.shared(value=np.dot(Z3.eval(), W4) + b4, name='A4')
Z4 = T.nnet.softmax(A4)

labels = np.argmax(Z4.eval(), axis=1)

pred_image = np.reshape(labels, (340, 610))

import matplotlib.pylab as plt
plt.imshow(pred_image)
plt.show()

labels_dict = {}
labels_dict['true_labels'] = y_test
labels_dict['pred_labels'] = labels
io.savemat('multi_data/PredictedLabels', labels_dict)


#Z3 = T.nnet.sigmoid(T.dot(Z2, W3) + b3)
#out = T.nnet.softmax(T.dot(Z3, W4) + b4)