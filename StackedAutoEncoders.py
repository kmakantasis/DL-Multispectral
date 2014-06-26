# -*- coding: utf-8 -*-

import theano.tensor as T
import theano
from theano.tensor.shared_randomstreams import RandomStreams
from MLP import HiddenLayer
from AutoEncoder import AutoEncoder
from LogisticRegression import LogisticRegression


class StackedAutoEncoders(object):
    def __init__(self, np_rng, theano_rng=None, n_ins=784, hidden_layer_sizes=[500, 500], n_outs=10):
        
        self.sigmoid_layers = []
        self.dA_layers = []
        self.params = []
        self.n_layers = len(hidden_layer_sizes)
        
        assert self.n_layers > 0

        if not theano_rng:
            theano_rng = RandomStreams(np_rng.randint(2 ** 30))
     
        self.x = T.matrix('x') 
        self.y = T.ivector('y') 
        
        for i in xrange(self.n_layers):
            if i == 0:
                n_in = n_ins
                layer_input = self.x
            else:
                n_in = hidden_layer_sizes[i-1]
                layer_input = self.sigmoid_layers[-1].output

            n_out = hidden_layer_sizes[i]            
            
            sigmoid_layer = HiddenLayer(np_rng, layer_input, n_in, n_out, activation=T.nnet.sigmoid)
            self.sigmoid_layers.append(sigmoid_layer)
            
            self.params.extend(sigmoid_layer.params)
            
            dA_layer = AutoEncoder(np_rng, n_in, n_out, theano_rng=theano_rng, input=layer_input, 
                                   W=sigmoid_layer.W, b_hid=sigmoid_layer.b)
            self.dA_layers.append(dA_layer)
            
        self.log_layer = LogisticRegression(self.sigmoid_layers[-1].output, self.y, hidden_layer_sizes[-1], n_outs)
        self.params.extend(self.log_layer.params)

        self.finetune_cost = self.log_layer.negative_log_likelihood()
        self.errors = self.log_layer.errors()        
        
        
    def pretraining_functions(self, train_set_x, batch_size):
        
        index = T.lscalar(name='index')
        corruption_level = T.scalar('corruption')
        learning_rate = T.scalar('lr')
        
        batch_begin = index * batch_size
        batch_end = batch_begin + batch_size
        
        pretrain_fns = []
        
        for dA in self.dA_layers:
            cost, updates = dA.get_cost_updates(learning_rate, corruption_level)
            fn =  theano.function(inputs=[index, theano.Param(corruption_level, default=0.2), theano.Param(learning_rate, default=0.1)],
                                          outputs=[cost],
                                          updates = updates,
                                          givens={self.x:train_set_x[batch_begin:batch_end]})
                
            pretrain_fns.append(fn)
            
        return pretrain_fns
        
        
    def finetuning_functions(self, datasets, batch_size, learning_rate):
        
        (train_set_x, train_set_y) = datasets[0]
        (valid_set_x, valid_set_y) = datasets[1]
        (test_set_x, test_set_y) = datasets[2]
        
        n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] / batch_size
        n_test_batches = test_set_x.get_value(borrow=True).shape[0] / batch_size
        
        index = T.lscalar('index')
        
        gparams = T.grad(self.finetune_cost, self.params)
        updates = []
        for param, gparam in zip(self.params, gparams):
            updates.append((param, param - gparam * learning_rate))
            
        train_fn = theano.function(inputs=[index],
                                   outputs=self.finetune_cost,
                                   updates=updates,
                                   givens={self.x: train_set_x[index * batch_size: (index+1) * batch_size],
                                           self.y: train_set_y[index * batch_size: (index+1) * batch_size]})

        test_score_i = theano.function(inputs=[index], 
                                       outputs=self.errors,
                                       givens={self.x: test_set_x[index * batch_size: (index+1) * batch_size],
                                               self.y: test_set_y[index * batch_size: (index+1) * batch_size]})

        valid_score_i = theano.function(inputs=[index], 
                                        outputs=self.errors,
                                        givens={self.x: valid_set_x[index * batch_size: (index+1) * batch_size],
                                                self.y: valid_set_y[index * batch_size: (index+1) * batch_size]})

        def valid_score():
            return [valid_score_i(i) for i in xrange(n_valid_batches)]

        def test_score():
            return [test_score_i(i) for i in xrange(n_test_batches)]

        return train_fn, valid_score, test_score   
            
            
            
            
            
            