# -*- coding: utf-8 -*-
import numpy as np
import time
import sys
import os
import theano
import theano.tensor as T
import LoadData
import ConvPoolLayer
import MultiLayerPerceptron
import LogisticLayer

def test_cnn(dataset_matrix_r, label_vector_r, learning_rate=0.1, n_epochs=120, nkerns=[30, 90], batch_size=250):
    
    # Load dataset
    datasets = LoadData.load_data_multi(dataset_matrix_r, label_vector_r)

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] / batch_size
    n_test_batches = test_set_x.get_value(borrow=True).shape[0] / batch_size

    # Construct the model
    print '... building the model'

    index = T.lscalar()  
    x = T.matrix('x')  
    y = T.ivector('y')  

    rng = np.random.RandomState(1234)
    
    layer0_input = x.reshape((batch_size, 5, 5, 10))
    layer0_input = layer0_input.dimshuffle(0, 3, 1, 2)

    layer0 = ConvPoolLayer.ConvPoolLayer(rng, 
                                         layer0_input, 
                                         filter_shape=(nkerns[0], 10, 3, 3),
                                         image_shape=(batch_size, 10, 5, 5))
                                        
    layer1 = ConvPoolLayer.ConvPoolLayer(rng,
                                         layer0.output,
                                         filter_shape=(nkerns[1], nkerns[0], 3, 3),
                                         image_shape=(batch_size, nkerns[0], 3, 3))
                                         
    layer3 = MultiLayerPerceptron.HiddenLayer(rng, 
                                              layer1.output.flatten(2),
                                              nkerns[1], 
                                              120, 
                                              activation=T.tanh)
                                                                                    
    layer5 = LogisticLayer.LogisticLayer(layer3.output, 120, 9)
    
    cost = layer5.negative_log_likelihood(y)
    
    # Function to train the model
    params = layer5.params + layer3.params + layer1.params +layer0.params
    gparams = T.grad(cost, params)
    updates = [(param, param - learning_rate * gparam) for param, gparam in zip(params, gparams)]
    train_model = theano.function(inputs=[index],
                                  outputs=[cost],
                                  updates=updates,
                                  givens={x:train_set_x[index * batch_size: (index+1) * batch_size],
                                          y:train_set_y[index * batch_size: (index+1) * batch_size]})
                                          
    # Functions to test and validate the model
    valid_model = theano.function(inputs=[index],
                                  outputs=[layer5.errors(y)],
                                  givens={x:valid_set_x[index * batch_size: (index+1) * batch_size],
                                          y:valid_set_y[index * batch_size: (index+1) * batch_size]})
                                          
    test_model = theano.function(inputs=[index],
                                 outputs=[layer5.errors(y)],
                                 givens={x:test_set_x[index * batch_size: (index+1) * batch_size],
                                         y:test_set_y[index * batch_size: (index+1) * batch_size]})
                                         
    print '... training the model'
    patience = 10000  
    patience_increase = 2  
    improvement_threshold = 0.995  
    validation_frequency = min(n_train_batches, patience / 2)
                                  
    best_validation_loss = np.inf
    best_iter = 0
    test_score = 0.
    start_time = time.clock()

    epoch = 0
    done_looping = False

    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in xrange(n_train_batches):

            iter = (epoch - 1) * n_train_batches + minibatch_index

            if iter % 100 == 0:
                print 'training @ iter = ', iter
            
            train_model(minibatch_index)

            if (iter + 1) % validation_frequency == 0:

                validation_losses = [valid_model(i) for i in xrange(n_valid_batches)]
                this_validation_loss = np.mean(validation_losses)
                print('epoch %i, minibatch %i/%i, validation error %f %%' %  (epoch, 
                                                                              minibatch_index + 1, 
                                                                              n_train_batches,
                                                                              this_validation_loss * 100.))

                if this_validation_loss < best_validation_loss:

                    if this_validation_loss < best_validation_loss *  improvement_threshold:
                        patience = max(patience, iter * patience_increase)

                    best_validation_loss = this_validation_loss
                    best_iter = iter

                    test_losses = [
                        test_model(i)
                        for i in xrange(n_test_batches)
                    ]
                    test_score = np.mean(test_losses)
                    print(('     epoch %i, minibatch %i/%i, test error of '
                           'best model %f %%') %
                          (epoch, minibatch_index + 1, n_train_batches,
                           test_score * 100.))

            if patience <= iter:
                done_looping = True
                break

    end_time = time.clock()
    print('Optimization complete.')
    print('Best validation score of %f %% obtained at iteration %i, '
          'with test performance %f %%' %
          (best_validation_loss * 100., best_iter + 1, test_score * 100.))
    print >> sys.stderr, ('The code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.))
                          
    return params
    
    

if __name__ == '__main__':
    dataset_matrix, label_vector, dataset_matrix_r, label_vector_r = LoadData.preprocess_data()
    params = test_cnn(dataset_matrix_r, label_vector_r)