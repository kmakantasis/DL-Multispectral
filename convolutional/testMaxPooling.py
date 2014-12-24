# -*- coding: utf-8 -*-
import numpy as np
import theano
import theano.tensor as T
from theano.tensor.signal import downsample

input = T.dmatrix('input')
maxpool_shape = (2, 2)
pool_out = downsample.max_pool_2d(input, maxpool_shape, ignore_border=True)
f = theano.function([input],pool_out)

invals = np.random.RandomState(1).rand(6, 6)
print 'With ignore_border set to True:'
print 'invals[0, 0, :, :] =\n', invals
print 'output[0, 0, :, :] =\n', f(invals)

pool_out = downsample.max_pool_2d(input, maxpool_shape, ignore_border=False)
f = theano.function([input],pool_out)
print 'With ignore_border set to False:'
print 'invals[1, 0, :, :] =\n ', invals
print 'output[1, 0, :, :] =\n ', f(invals)
