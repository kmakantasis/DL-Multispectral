# -*- coding: utf-8 -*-

import numpy as np
import time
import os 
import sys
from loadDataset import load_data, load_multi
from StackedAutoEncoders import StackedAutoEncoders

def StackedAutoEncoders_demo(finetune_lr=0.1, pretraining_epochs=10, pretrain_lr=0.001, training_epochs=1000, dataset='mnist.pkl.gz', batch_size=20, pretrain_flag=True):
    
    datasets = load_multi()

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    n_train_batches = train_set_x.get_value(borrow=True).shape[0]
    n_train_batches /= batch_size
    
    numpy_rng = np.random.RandomState(89677)
    print '... building the model'
    
    sda = StackedAutoEncoders(numpy_rng, n_ins=103, hidden_layer_sizes=[103, 103, 103, 103, 103, 103], n_outs=9)
  
  
    #########################
    # PRETRAINING THE MODEL #
    #########################
    if pretrain_flag == True:
        print '... getting the pretraining functions'
        pretraining_fns = sda.pretraining_functions(train_set_x, batch_size)
        start_time = time.clock()
    
        print '... pre-training the model'
    
        corruption_levels = [.1, .2, .3, .3, .3, .3]
        for i in xrange(sda.n_layers):
        
            for epoch in xrange(pretraining_epochs):
            
                c = []
                for batch_index in xrange(n_train_batches):
                    c.append(pretraining_fns[i](index=batch_index,
                             corruption=corruption_levels[i],
                             lr=pretrain_lr))
                print 'Pre-training layer %i, epoch %d, cost ' % (i, epoch),
                print np.mean(c)

        end_time = time.clock()
        print >> sys.stderr, ('The pretraining code for file ' +
                              os.path.split(__file__)[1] +
                              ' ran for %.2fm' % ((end_time - start_time) / 60.))
                              
    else:
        print '... pretraining skipped'
  
    ########################
    # FINETUNING THE MODEL #
    ########################
  
    print '... getting the finetuning functions'
    train_fn, validate_model, test_model = sda.finetuning_functions(datasets, batch_size, finetune_lr)

    print '... finetunning the model'
    patience = 10 * n_train_batches  
    patience_increase = 2. 
    improvement_threshold = 0.995  
    validation_frequency = min(n_train_batches, patience / 2)
                                  
    best_validation_loss = np.inf
    test_score = 0.
    start_time = time.clock()

    done_looping = False
    epoch = 0

    while (epoch < training_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in xrange(n_train_batches):
            train_fn(minibatch_index)
            iter = (epoch - 1) * n_train_batches + minibatch_index

            if (iter + 1) % validation_frequency == 0:
                validation_losses = validate_model()
                this_validation_loss = np.mean(validation_losses)
                print('epoch %i, minibatch %i/%i, validation error %f %%' %
                      (epoch, minibatch_index + 1, n_train_batches,
                       this_validation_loss * 100.))

                if this_validation_loss < best_validation_loss:
                    
                    if (this_validation_loss < best_validation_loss *
                        improvement_threshold):
                        patience = max(patience, iter * patience_increase)
                        
                    best_validation_loss = this_validation_loss

                    test_losses = test_model()
                    test_score = np.mean(test_losses)
                    print(('     epoch %i, minibatch %i/%i, test error of '
                           'best model %f %%') %
                          (epoch, minibatch_index + 1, n_train_batches,
                           test_score * 100.))

            if patience <= iter:
                done_looping = True
                break

    end_time = time.clock()
    print(('Optimization complete with best validation score of %f %%,'
           'with test performance %f %%') %
                 (best_validation_loss * 100., test_score * 100.))
    print >> sys.stderr, ('The training code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.))
  
    
if __name__ == '__main__':
    StackedAutoEncoders_demo()