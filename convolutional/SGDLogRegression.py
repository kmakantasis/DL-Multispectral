# -*- coding: utf-8 -*-
import numpy as np
import time
import sys
import os
import theano
import theano.tensor as T
import LoadData
import LogisticLayer


def sgd_optimization_mnist(learning_rate=0.13, n_epochs=1000, dataset='mnist.pkl.gz', batch_size=600):
    """ Stochastic Gradient Descent for logistic Regression """
    
    # Load dataset and create batches
    datasets = LoadData.load_data(dataset)
    
    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] / batch_size
    n_test_batches = test_set_x.get_value(borrow=True).shape[0] / batch_size


    # Construct the model
    print 'Building the model ... '
        
    index = T.iscalar('index')    
    x = T.dmatrix('x')
    y = T.ivector('y')
    
    classifier = LogisticLayer.LogisticLayer(x, 28*28, 10)
    cost = classifier.negative_log_likelihood(y)
    
    # Function to train the model
    gW = T.grad(cost, classifier.W)
    gb = T.grad(cost, classifier.b)
    updates = [(classifier.W, classifier.W - learning_rate * gW),
               (classifier.b, classifier.b - learning_rate * gb)]
    train_model = theano.function(inputs=[index], 
                                  outputs=[cost],
                                  updates=updates,
                                  givens={x:train_set_x[index * batch_size: (index + 1) * batch_size],
                                          y: train_set_y[index * batch_size: (index + 1) * batch_size]})
                                          
    # Functions to test and validate the model
    valid_model = theano.function(inputs=[index],
                                  outputs=[classifier.errors(y)],
                                  givens={x:valid_set_x[index * batch_size: (index+1) * batch_size],
                                          y:valid_set_y[index * batch_size: (index+1) * batch_size]})
                                          
    test_model = theano.function(inputs=[index],
                                 outputs=[classifier.errors(y)],
                                 givens={x:test_set_x[index * batch_size: (index+1) * batch_size],
                                         y:test_set_y[index * batch_size: (index+1) * batch_size]})
                                         
    # Train the model
    print 'Training the model ...'
    
    patience = 5000  
    patience_increase = 2  
    improvement_threshold = 0.995  
    validation_frequency = min(n_train_batches, patience / 2)
                                  
    best_validation_loss = np.inf
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
                validation_losses = [valid_model(i) for i in xrange(n_valid_batches)]
                this_validation_loss = np.mean(validation_losses)

                print('epoch %i, minibatch %i/%i, validation error %f %%' % (epoch,
                                                                             minibatch_index + 1,
                                                                             n_train_batches,
                                                                             this_validation_loss * 100.) )

                if this_validation_loss < best_validation_loss:
                    if this_validation_loss < best_validation_loss * improvement_threshold:
                        patience = max(patience, iter * patience_increase)

                    best_validation_loss = this_validation_loss

                    test_losses = [test_model(i) for i in xrange(n_test_batches)]
                    test_score = np.mean(test_losses)

                    print( ('     epoch %i, minibatch %i/%i, test error of best model %f %%') % (epoch,
                                                                                                 minibatch_index + 1,
                                                                                                 n_train_batches,
                                                                                                 test_score * 100.))

            if patience <= iter:
                done_looping = True
                break

    end_time = time.clock()
    print(
        (
            'Optimization complete with best validation score of %f %%,'
            'with test performance %f %%'
        )
        % (best_validation_loss * 100., test_score * 100.)
    )
    print 'The code run for %d epochs, with %f epochs/sec' % (
        epoch, 1. * epoch / (end_time - start_time))
    print >> sys.stderr, ('The code for file ' + os.path.split(__file__)[1] + ' ran for %.1fs' % ((end_time - start_time)))

if __name__ == '__main__':
    sgd_optimization_mnist()
                                          
    
    