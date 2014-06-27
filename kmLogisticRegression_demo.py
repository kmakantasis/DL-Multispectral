# -*- coding: utf-8 -*-

import os
import sys
import time
import numpy
import theano
import theano.tensor as T
from LogisticRegression import LogisticRegression
from loadDataset import load_data, load_multi


def LogisticRegression_demo(learning_rate=0.13, n_epochs=1000, dataset='mnist.pkl.gz', batch_size=600):
    datasets = load_multi()

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

    classifier = LogisticRegression(x, y, n_in=103, n_out=9)
    
    test_model = theano.function(inputs=[index],
                                 outputs=classifier.errors(),
                                 givens={x: test_set_x[index * batch_size: (index + 1) * batch_size],
                                         y: test_set_y[index * batch_size: (index + 1) * batch_size]})
                                         
    validate_model = theano.function(inputs=[index],
                                     outputs=classifier.errors(),
                                     givens={x: valid_set_x[index * batch_size:(index + 1) * batch_size],
                                             y: valid_set_y[index * batch_size:(index + 1) * batch_size]})
                                             
    cost, updates = classifier.get_cost_updates(learning_rate=learning_rate)
    
    train_model = theano.function(inputs=[index],
                                  outputs=cost,
                                  updates=updates,
                                  givens={x: train_set_x[index * batch_size:(index + 1) * batch_size],
                                          y: train_set_y[index * batch_size:(index + 1) * batch_size]})
                                          
    print '... training the model'
    
    patience = 5000  
    patience_increase = 2  
                                  
    improvement_threshold = 0.995  
    validation_frequency = min(n_train_batches, patience / 2)

    best_validation_loss = numpy.inf
    test_score = 0.
    start_time = time.clock()

    done_looping = False
    epoch = 0
    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in xrange(n_train_batches):

            train_model(minibatch_index)
            iter = (epoch - 1) * n_train_batches + minibatch_index

            if (iter + 1) % validation_frequency == 0:
                validation_losses = [validate_model(i)
                                     for i in xrange(n_valid_batches)]
                this_validation_loss = numpy.mean(validation_losses)

                print('epoch %i, minibatch %i/%i, validation error %f %%' % \
                    (epoch, minibatch_index + 1, n_train_batches,
                    this_validation_loss * 100.))

                if this_validation_loss < best_validation_loss:
                    if this_validation_loss < best_validation_loss *  \
                       improvement_threshold:
                        patience = max(patience, iter * patience_increase)

                    best_validation_loss = this_validation_loss

                    test_losses = [test_model(i)
                                   for i in xrange(n_test_batches)]
                    test_score = numpy.mean(test_losses)

                    print(('     epoch %i, minibatch %i/%i, test error of best'
                       ' model %f %%') %
                        (epoch, minibatch_index + 1, n_train_batches,
                         test_score * 100.))

            if patience <= iter:
                done_looping = True
                break

    end_time = time.clock()
    print(('Optimization complete with best validation score of %f %%,'
           'with test performance %f %%') %
                 (best_validation_loss * 100., test_score * 100.))
    print 'The code run for %d epochs, with %f epochs/sec' % (
        epoch, 1. * epoch / (end_time - start_time))
    print >> sys.stderr, ('The code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.1fs' % ((end_time - start_time)))

if __name__ == '__main__':
    LogisticRegression_demo()

    