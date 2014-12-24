# -*- coding: utf-8 -*-
import numpy as np
import theano
import theano.tensor as T


class LogisticLayer(object):
    """ Layer for conducting Logistic regression """
    
    def __init__(self, input, n_in, n_out, W=None, b=None):
        
        # Initialize weight and biases
        if W==None:
            self.W = theano.shared(value=np.zeros( (n_in, n_out), dtype=theano.config.floatX), name='W', borrow=True )
        else:
            self.W = theano.shared(value=W, name='W', borrow=False )
            
        if b==None:
            self.b = theano.shared(value=np.zeros( (n_out,), dtype=theano.config.floatX), name='b', borrow=True )
        else:
            self.b = theano.shared(value=b, name='b', borrow=False )
        
        self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)
        
        self.params = [self.W, self.b]

        
    def negative_log_likelihood(self, y):
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])
        
        
    def errors(self, y):
        if y.ndim != self.y_pred.ndim:
            raise TypeError('y should have the same shape as self.y_pred', ('y', y.type, 'y_pred', self.y_pred.type) )
        
        if y.dtype.startswith('int'):
            return T.mean(T.neq(self.y_pred, y))
        else:
            raise NotImplementedError()
