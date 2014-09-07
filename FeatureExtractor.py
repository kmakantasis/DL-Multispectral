# -*- coding: utf-8 -*-

import theano
import theano.tensor as T
import sys


class FeatureExtractor(object):
    def __init__(self, params, input, activation):
        
        if len(params) == 0:
            print >> sys.stderr, ('Params argument cannot be empty')
            return 0
        
        self.input = input
        self.params = params
        self.activation = activation
        self.output = None
        
    
    def ExtractFeatures(self):
        
        n_layers = len(self.params) / 2
        
        x = theano.shared(value=self.input, name='x', borrow=True)
        output = x
        
        for i in range(n_layers):
            index = i * 2
            W = theano.shared(value=self.params[index], name='W')
            b = theano.shared(value=self.params[index+1], name='b')
            output = self.activation(T.dot(output, W) + b)
            
        self.output = output.eval()
            
        