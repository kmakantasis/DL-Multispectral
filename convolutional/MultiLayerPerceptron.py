# -*- coding: utf-8 -*-
import numpy as np
import theano
import theano.tensor as T
import LogisticLayer

class HiddenLayer(object):
    """ Hidden layer object. """
    def __init__(self, rng, input, n_in, n_out, W=None, b=None, activation=T.tanh):
        
        self.input = input
        
        # Weights and biases initialization
        if W == None:
            W_bound = np.sqrt(6. / (n_in + n_out))
            W_values = np.asarray( rng.uniform(low=-W_bound, high=W_bound, size=(n_in, n_out)), dtype=theano.config.floatX)
            if activation == T.nnet.sigmoid:
                W_values = 4 * W_values
                
            self.W = theano.shared(value=W_values, name='W', borrow=True)
        else:
            self.W = theano.shared(value=W, name='W', borrow=False)
            
            
        if b == None:
            b_values = np.zeros((n_out, ), dtype=theano.config.floatX)
            self.b = theano.shared(value=b_values, name='b', borrow=True)     
        else:
            self.b = theano.shared(value=b, name='b', borrow=False)    
 
        # Network output
        self.output = activation(T.dot(input, self.W) + self.b)

        self.params = [self.W, self.b]


class MLP(object):
    """ Multilayered perceptron object. """ 
    def __init__(self, rng, input, n_in, n_hidden, n_out):
        # Construction of hidden and logistic layers 
        self.hiddenLayer = HiddenLayer(rng, input, n_in, n_hidden, W=None, b=None, activation=T.tanh) 
        self.logLayer = LogisticLayer.LogisticLayer(self.hiddenLayer.output, n_hidden, n_out)
        
        # Regularization techniques
        self.L1 = abs(self.hiddenLayer.W).sum() + abs(self.logLayer.W).sum()
        self.L2 = (self.hiddenLayer.W ** 2).sum() + (self.logLayer.W ** 2).sum()

        # Cost function and errors
        self.negative_log_likelihood = self.logLayer.negative_log_likelihood
        self.errors = self.logLayer.errors
        
        self.params = self.hiddenLayer.params + self.logLayer.params
        
        
        