# -*- coding: utf-8 -*-

import os
import sys
import time
import numpy
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
from utils import tile_raster_images
import PIL.Image
from AutoEncoder import AutoEncoder
from loadDataset import load_data


def AutoEncoder_demo(learning_rate=0.1, training_epochs=2, dataset='mnist.pkl.gz', batch_size=20, output_folder='dA_plots'):
   
    datasets = load_data(dataset)
    train_set_x, train_set_y = datasets[0]
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
    index = T.lscalar()    # index to a [mini]batch
    x = T.matrix('x')  
    
    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)
    os.chdir(output_folder)
 
   
    #####################################
    # BUILDING THE MODEL CORRUPTION 0% #
    #####################################    
    rng = numpy.random.RandomState(123)
    theano_rng = RandomStreams(rng.randint(2 ** 30))

    da = AutoEncoder(np_rng=rng, theano_rng=theano_rng, input=x, n_vis=28 * 28, n_hid=500)
    cost, updates = da.get_cost_updates(corruption_level=0., learning_rate=learning_rate)
    train_da = theano.function(inputs=[index], 
                               outputs=[cost], 
                               updates=updates,
                               givens={x: train_set_x[index * batch_size: (index + 1) * batch_size]})

    start_time = time.clock()
    for epoch in xrange(training_epochs):
        # go through trainng set
        c = []
        for batch_index in xrange(n_train_batches):
            c.append(train_da(batch_index))

        print 'Training epoch %d, cost ' % epoch, numpy.mean(c)
    end_time = time.clock()

    training_time = (end_time - start_time)
    print >> sys.stderr, ('The no corruption code ran for %.2fm' % ((training_time) / 60.))
    image = PIL.Image.fromarray(tile_raster_images(X=da.W.get_value(borrow=True).T,
                                                   img_shape=(28, 28), tile_shape=(10, 10),
                                                   tile_spacing=(1, 1)))
    image.save('filters_corruption_0.jpg')


    #####################################
    # BUILDING THE MODEL CORRUPTION 30% #
    #####################################
    rng = numpy.random.RandomState(123)
    theano_rng = RandomStreams(rng.randint(2 ** 30))

    da = AutoEncoder(np_rng=rng, theano_rng=theano_rng, input=x, n_vis=28 * 28, n_hid=500)
    cost, updates = da.get_cost_updates(corruption_level=0.3, learning_rate=learning_rate)
    train_da = theano.function(inputs=[index], 
                               outputs=[cost], 
                               updates=updates,
                               givens={x: train_set_x[index * batch_size:(index + 1) * batch_size]})

    start_time = time.clock()   
    for epoch in xrange(training_epochs):
        c = []
        for batch_index in xrange(n_train_batches):
            c.append(train_da(batch_index))

        print 'Training epoch %d, cost ' % epoch, numpy.mean(c)
    end_time = time.clock()
    
    training_time = (end_time - start_time)
    print >> sys.stderr, ('The 30 percent corruption code ran for %.2fm' % ((training_time) / 60.))
    image = PIL.Image.fromarray(tile_raster_images(X=da.W.get_value(borrow=True).T,
                                                   img_shape=(28, 28), tile_shape=(10, 10),
                                                   tile_spacing=(1, 1)))
    image.save('filters_corruption_30.jpg')

    os.chdir('../')


if __name__ == '__main__':
    AutoEncoder_demo()
    