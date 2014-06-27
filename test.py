# -*- coding: utf-8 -*-


import theano
import theano.tensor as T
from loadDataset import load_data
from LogisticRegression import LogisticRegression


batch_size = 20
datasets = load_data(dataset)

train_set_x, train_set_y = datasets[0]
valid_set_x, valid_set_y = datasets[1]
test_set_x, test_set_y = datasets[2]

n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] / batch_size
n_test_batches = test_set_x.get_value(borrow=True).shape[0] / batch_size
    
print '... building the model'

    
index = T.lscalar()  


x = T.matrix('x') 
y = T.ivector('y') 

classifier = LogisticRegression(x, y, n_in=784, n_out=10)

#cost = T.log(classifier.p_y_given_x)[T.arange(y.shape[0]), y]
cost = T.log(classifier.p_y_given_x)[T.arange(y.shape[0]), y]


train_model = theano.function(inputs=[index],
                                  outputs=cost,
                                  givens={x: train_set_x[index * batch_size:(index + 1) * batch_size],
                                          y: train_set_y[index * batch_size:(index + 1) * batch_size]})
                                          
                                          
loss = train_model(1)