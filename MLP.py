# -*- coding: utf-8 -*-

import numpy as np
import theano
import theano.tensor as T
from LogisticRegression import LogisticRegression

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
                
            W = theano.shared(value=initial_W, name='W', borrow=True)
            
            
        if b is None:
            initial_b = np.asarray(np.zeros(shape=(n_out,), dtype=theano.config.floatX))
            b = theano.shared(value=initial_b, name='b', borrow=True)
            
        self.W = W
        self.b = b
        self.x = input
        
        output = T.dot(self.x, self.W) + b
        self.output = (output if activation is None else activation(output))
        
        self.params = [ self.W, self.b]
        
        
class MLP(object):
    def __init__(self, rng, input, y, n_in, n_hidden, n_out, activation=T.nnet.sigmoid):   
        """
            Theory
            ++++++
            An MLP consists of one or more hidden layers and one output layer. Hidden layers use 
            as activation function a non-linear function such as ``theano.tensor.nnet.sigmoid`` or
            ``theano.tensor.tahn``, while the output layer uses the softmax algorithm.
                        
            Function definition
            +++++++++++++++++++
            
            .. py:function:: __init__(rng, input, y, n_in, n_hidden, n_out, activation=T.nnet.sigmoid)

               Initializes the parameters of a multi-layered perceptron model.
             
               :param numpy.random.RandomState np_rng: number random generator used to generate weights
               :param theano.tensor.TensorType input: a symbolic description of the input
               :param theano.tensor.TensorType y: a symbolic description of the output
               :param int n_in: input dimension
               :param int hidden: number of neurons at the hidden layer  
               :param int n_out: number of neurons at the last logistic layer
        """
        
        self.hidden_layer = HiddenLayer(rng, input, n_in, n_hidden, activation=activation)
        self.log_layer = LogisticRegression(self.hidden_layer.output, y, n_hidden, n_out)
        
        self.L1 = abs(self.hidden_layer.W).sum() + abs(self.log_layer.W).sum()
        self.L2 = (self.hidden_layer.W ** 2).sum() + (self.log_layer.W ** 2).sum()
        
        self.negative_log_likelihood = self.log_layer.negative_log_likelihood
        self.errors = self.log_layer.errors
        
        self.params = self.hidden_layer.params + self.log_layer.params
        
        
    def get_cost_updates(self, learning_rate, L1_reg, L2_reg):
        """
            Function definition
            +++++++++++++++++++
            
            .. py:function:: get_cost_updates(learning_rate, L1_reg, L2_reg)
            
               Calculates the derivatives of cost function 
               with respect to :math:`W_h, b_h, W_l, b_l`. Then, it uses the learning rate to calculate the 
               updates for the model parameters that are going to be used with a gradient based
               optimization method during model training.

               :param float learning_rate: learning rate for a gradient based optimization algorithm.
               :param float L1_reg: weight of L1 regularization term.
               :param float L2_reg: weight of L2 regularization term.
               :return: the evaluation of the cost function and the updates for the MLP 
                        model parameters that are going to be used by gradient based optimization 
                        during training phase.
               :rtype: float - list of floats
        """
        
        cost = self.negative_log_likelihood() + L1_reg * self.L1 + L2_reg * self.L2     
        gparams = T.grad(cost, self.params)
        updates = []
        
        for param, gparam in zip(self.params, gparams):
            updates.append((param, param - learning_rate * gparam))
        
        return (cost, updates)
        