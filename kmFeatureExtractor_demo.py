# -*- coding: utf-8 -*-

import scipy.io as io
import theano.tensor as T
from sklearn.cluster import KMeans
import matplotlib.pylab as plt
from FeatureExtractor import FeatureExtractor


weights = io.loadmat('multi_data/TrainedWeightsNoFinetuning.mat')

params2 = [weights['W1'], weights['b1'][0,:], weights['W2'], 
           weights['b2'][0,:], weights['W3'], weights['b3'][0,:]]
           
hypercube = io.loadmat('multi_data/hypercube.mat')

fe_input = hypercube['Hypercube']
fe_input = fe_input.reshape((-1,103))

fe = FeatureExtractor(params2, fe_input, T.nnet.sigmoid)
fe.ExtractFeatures()

features = fe.output

print 'Clustering has started...'

kmeans = KMeans(init='k-means++', n_clusters=9, n_init=5, precompute_distances=False)
kmeans.fit(features)

centroids = kmeans.cluster_centers_
labels = kmeans.labels_

pred_img = labels.reshape((610, 340))

plt.imshow(pred_img)