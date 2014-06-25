# -*- coding: utf-8 -*-

import numpy as np
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams


x_num = np.ones((2,10))
z_num = 2*np.ones((2,10))

x = T.dmatrix('x')
z = T.dmatrix('z')

L = T.sum(((x - z)**2)/2., axis=1)

cost = T.mean(L)


f = theano.function(inputs=[x,z], outputs=[cost])

theano_rng = RandomStreams(np.random.randint(2 ** 30))
asd = theano_rng.binomial(size=x.shape, n=1, p=0.8)

f2 = theano.function(inputs=[x], outputs=[asd])