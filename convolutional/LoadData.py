# -*- coding: utf-8 -*-
import numpy as np
from sklearn.decomposition import RandomizedPCA
import theano
import theano.tensor as T
import scipy.io as sio


def preprocess_data():
    datasets = sio.loadmat('../multi_data/Hyper_01_Urban.mat')
    hypercube = datasets['Hypercube']

    datasets = sio.loadmat('../multi_data/Hyper_01_Urban_GroundTruth.mat')
    ground_truth = datasets['Ground_Truth']

    del datasets

    hypercube_1D = np.reshape(hypercube, (-1, hypercube.shape[2]))
    rpca = RandomizedPCA(n_components=10, whiten=True)
    hypercube_1D_reduced = rpca.fit_transform(hypercube_1D)
    hypercube_reduced = np.reshape(hypercube_1D_reduced, (hypercube.shape[0], hypercube.shape[1], -1))

    print rpca.explained_variance_ratio_.sum()

    window_sz = 5
    window_pad = 2
    dataset_matrix_size = ((hypercube_reduced.shape[0]-window_pad) * (hypercube_reduced.shape[1]-window_pad), window_sz, window_sz, hypercube_reduced.shape[2])
    dataset_matrix = np.zeros(dataset_matrix_size)
    label_vector = np.zeros((dataset_matrix.shape[0],))

    data_index = 0
    for r in range(hypercube_reduced.shape[0]):
        if r < window_pad or r > hypercube_reduced.shape[0] - window_pad-1:
            continue
        for c in range(hypercube_reduced.shape[1]):
            if c < window_pad or c > hypercube_reduced.shape[1] - window_pad-1:
                continue
        
            patch = hypercube_reduced[r-window_pad:r+window_pad+1, c-window_pad:c+window_pad+1]
            dataset_matrix[data_index,:,:,:] = patch
            label_vector[data_index] = ground_truth[r,c]        
        
            data_index = data_index + 1
        

    dataset_matrix_r = dataset_matrix[label_vector>0,:,:,:]
    label_vector_r = label_vector[label_vector>0]

    rand_perm = np.random.permutation(label_vector_r.shape[0])
    dataset_matrix_r = dataset_matrix_r[rand_perm,:,:,:]
    label_vector_r = label_vector_r[rand_perm]
    
    label_vector_r = label_vector_r - 1.0
    
    return dataset_matrix, label_vector, dataset_matrix_r, label_vector_r



def load_data_multi(dataset_matrix_r, label_vector_r):
    
    def shared_dataset(start, end, data_x, data_y, borrow=True):
        x_ar = np.reshape(data_x[start:end], (-1, 250) )
        shared_x = theano.shared(np.asarray(x_ar, dtype=theano.config.floatX), borrow=borrow)
        shared_y = theano.shared(np.asarray(data_y[start:end], dtype=theano.config.floatX), borrow=borrow)
        
        return shared_x, T.cast(shared_y, 'int32')
        
    test_set_x, test_set_y = shared_dataset(31500, 41500, dataset_matrix_r, label_vector_r)
    valid_set_x, valid_set_y = shared_dataset(31500, 41500, dataset_matrix_r, label_vector_r)
    train_set_x, train_set_y = shared_dataset( 0, 31500, dataset_matrix_r, label_vector_r)

    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y), (test_set_x, test_set_y)]
    
    return rval
    
    

