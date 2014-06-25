# -*- coding: utf-8 -*-

import numpy as np
import theano
import theano.tensor as T

class HiddenLayer(object):
    def __init__(self, rng, input, n_in, n_out, W=None, b=None, activation=T.tanh):
        """
            Theory
            ++++++
            In an MLP, a hidden layer takes an input :math:`x` and transforms it to :math:`y` 
            using its activation function :math:`f()` according to the following relation:
            
            .. math::
            
                y = f(Wx + b)
                
            Activation functions can be ``theano.tensor.nnet.sigmoid``, ``theano.tensor.tanh`` or ``None``.
                        
            Function definition
            +++++++++++++++++++
            
            .. py:function:: __init__(rng, input, n_in, n_out, W=None, b=None, activation=T.tanh)

               Initializes the parameters of a hidden layer model.
             
               :param numpy.random.RandomState np_rng: number random generator used to generate weights
               :param theano.tensor.TensorType input: a symbolic description of the input
               :param int n_in: number of neurons at the layer before the hidden (current) layer
               :param int n_out: number of neurons at the hidden (current) layer           
        """        
        if W is None:
            initial_W = np.asarray(rng.uniform(low = -np.sqrt(6. / (n_in + n_out)),
                                               high=np.sqrt(6. / (n_in + n_out)),
                                               size=(n_in, n_out)), dtype=theano.config.floatX)
                                               
            if activation == T.nnet.sigmoid:
                initial_W = initial_W * 4
                
            W = initial_W
            
            
        if b is None:
            initial_b = np.asarray(np.zeros(shape=(n_out,), dtype=theano.config.floatX))
            b = initial_b
            
        self.W = W
        self.b = b
        self.x = input
        
        output = T.dot(self.x, self.W) + b
        self.output = (output if activation is None else activation(output))
        
        self.params = [ self.W, self.b]