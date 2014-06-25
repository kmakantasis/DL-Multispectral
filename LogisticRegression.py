# -*- coding: utf-8 -*-

import numpy as np
import theano
import theano.tensor as T


class LogisticRegression(object):
    def __init__(self, input, y, n_in, n_out):
        """
            Theory
            ++++++
            Logistic regression is a linear classifier. The parameters of a logistic regression
            model can be represented by the set :math:`params = \{W, b}`.
            
            The classification task, mathematically, can be expressed as:
            
            .. math::
            
                P(Y=i|x,W,b) = softmax(Wx+b) = \\frac{e^{W_i+b_i}}{\sum_j e^{W_j+b_j}}
                
            The prediction of the model is done as follows:
            
            .. math::
            
                p_{pred} = argmax_i P(Y=i|x,W,b)
                
                        
            Function definition
            +++++++++++++++++++
            
            .. py:function:: __init__(input, n_in, n_out)

               Initializes the parameters of a logistic regression model.
             
               :param theano.tensor.TensorType input: a symbolic description of the input
               :param theano.tensor.TensorType y: a symbolic description of the output
               :param int n_in: dimension of each input sample
               :param int n_out: number of classes           
        """
        
        self.W = theano.shared(value=np.zeros(shape=(n_in, n_out), dtype=theano.config.floatX), 
                               name='W', borrow=True)                             
        self.b = theano.shared(value=np.zeros(shape=(n_out,), dtype=theano.config.floatX),
                               name='b', borrow=True)                               
        self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)        
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)
        self.params = [self.W, self.b]        
        self.y = y
        
        
    def negative_log_likelihood(self):
        """
            Function definition
            +++++++++++++++++++   
            .. py:function:: negative_log_likelihood()
            
                Return the mean of the negative log-likelihood of the prediction
                of this model under a given target distribution.

                .. math::

                    -\\frac{1}{|\mathcal{D}|} \mathcal{L} (\{W,b\}, \mathcal{D}) =
                    -\\frac{1}{|\mathcal{D}|} \sum_{i=0}^{|\mathcal{D}|} \log(P(Y=y^{(i)}|x^{(i)}, W,b)) 
        """
        return -T.mean(T.log(self.p_y_given_x)[T.arange(self.y.shape[0]), self.y])
        
    
    def get_cost_updates(self, learning_rate):
        """
            Function definition
            +++++++++++++++++++
            
            .. py:function:: get_cost_updates(learning_rate)
            
               Calculates the derivatives of cost function 
               
               .. math::      
            
                   -\\frac{1}{|\mathcal{D}|} \mathcal{L} (\{W,b\}, \mathcal{D}) =
                    -\\frac{1}{|\mathcal{D}|} \sum_{i=0}^{|\mathcal{D}|} \log(P(Y=y^{(i)}|x^{(i)}, W,b)) 
                   
               with respect to :math:`W, b`. Then, it uses the learning rate to calculate the 
               updates for the model parameters that are going to be used with a gradient based
               optimization method during model training.

               :param float learning_rate: learning rate for a gradient based optimization algorithm.
               :return: the evaluation of the cost function and the updates for the autoencoder 
                        model parameters that are going to be used by gradient based optimization 
                        during training phase.
               :rtype: float - list of floats
        """
        
        cost = self.negative_log_likelihood()      
        gparams = T.grad(cost, self.params)
        updates = []
        
        for param, gparam in zip(self.params, gparams):
            updates.append((param, param - learning_rate * gparam))
        
        return (cost, updates)
        
        
    def errors(self):      
        """
            Return a float representing the number of errors in the minibatch
            over the total number of examples of the minibatch ; zero one
            loss over the size of the minibatch
        """
        return T.mean(T.neq(self.y_pred, self.y))
       