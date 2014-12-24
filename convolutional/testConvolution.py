# -*- coding: utf-8 -*-
import theano
import theano.tensor as T
import numpy as np
import pylab
from PIL import Image

rng = np.random.RandomState(23455)

input = T.tensor4(name='input')

# 9 different filters of size 9x9 applied on 3 channels
w_shp = (9,3,9,9) 
W = theano.shared(np.asarray(rng.uniform(low=-1.0 / np.sqrt(3*9*9),
                             high=1.0 / np.sqrt(3*9*9),
                             size=w_shp),
                             dtype=input.dtype), 
                 name ='W')

b_shp = (9,)
b = theano.shared(np.asarray(rng.uniform(low=-.5, high=.5, size=b_shp),
                             dtype=input.dtype), 
                  name ='b')
                  
conv_out = T.nnet.conv.conv2d(input, W)
output = T.nnet.sigmoid(conv_out + b.dimshuffle('x', 0, 'x', 'x'))

theanoConvolve = theano.function([input], output)

img = Image.open(open('images/3wolfmoon.jpg'))
img = np.asarray(img, dtype='float64') / 256.
img_ = img.swapaxes(0, 2).swapaxes(1, 2).reshape(1, 3, 639, 516)
filtered_img = theanoConvolve(img_)

pylab.subplot(2, 5, 1); pylab.axis('off'); pylab.imshow(img); pylab.gray();
pylab.subplot(2, 5, 2); pylab.axis('off'); pylab.imshow(filtered_img[0, 0, :, :])
pylab.subplot(2, 5, 3); pylab.axis('off'); pylab.imshow(filtered_img[0, 1, :, :])
pylab.subplot(2, 5, 4); pylab.axis('off'); pylab.imshow(filtered_img[0, 2, :, :])
pylab.subplot(2, 5, 5); pylab.axis('off'); pylab.imshow(filtered_img[0, 3, :, :])

pylab.subplot(2, 5, 6); pylab.axis('off'); pylab.imshow(filtered_img[0, 4, :, :])
pylab.subplot(2, 5, 7); pylab.axis('off'); pylab.imshow(filtered_img[0, 5, :, :])
pylab.subplot(2, 5, 8); pylab.axis('off'); pylab.imshow(filtered_img[0, 6, :, :])
pylab.subplot(2, 5, 9); pylab.axis('off'); pylab.imshow(filtered_img[0, 7, :, :])
pylab.subplot(2, 5, 10); pylab.axis('off'); pylab.imshow(filtered_img[0, 8, :, :])

pylab.show()