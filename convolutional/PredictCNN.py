# -*- coding: utf-8 -*-
import numpy as np
import scipy.io as sio
import theano
import theano.tensor as T
import LoadData
import ConvPoolLayer
import MultiLayerPerceptron
import LogisticLayer


def predict_cnn(nkerns=[20, 40, 60], batch_size=200): 
    
    # Load dataset
    datasets = LoadData.load_predict('VisionFeatures/dct12')

    #train_set_x, train_set_y = datasets[0]
    #valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    #n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
    #n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] / batch_size
    n_test_batches = test_set_x.get_value(borrow=True).shape[0] / batch_size
    
    weights = sio.loadmat('weights20')
    layer0_W = weights['layer0_W']
    layer0_b = weights['layer0_b']
    layer0_b = np.reshape(layer0_b, (layer0_b.shape[1],))
    layer1_W = weights['layer1_W']
    layer1_b = weights['layer1_b']
    layer1_b = np.reshape(layer1_b, (layer1_b.shape[1],))
    layer2_W = weights['layer2_W']
    layer2_b = weights['layer2_b']
    layer2_b = np.reshape(layer2_b, (layer2_b.shape[1],))
    layer3_W = weights['layer3_W']
    layer3_b = weights['layer3_b']
    layer3_b = np.reshape(layer3_b, (layer3_b.shape[1],))
    layer5_W = weights['layer5_W']
    layer5_b = weights['layer5_b']
    layer5_b = np.reshape(layer5_b, (layer5_b.shape[1],))


    # Construct the model
    print '... building the model'

    index = T.lscalar()  
    x = T.matrix('x')  
    y = T.ivector('y')  

    rng = np.random.RandomState(1234)
    
    layer0_input = x.reshape((batch_size, 72, 88, 1))
    layer0_input = layer0_input.dimshuffle(0, 3, 1, 2)

    layer0 = ConvPoolLayer.ConvPoolLayer(rng, 
                                         layer0_input, 
                                         filter_shape=(nkerns[0], 1, 9, 9),
                                         image_shape=(batch_size, 1, 72, 88),
                                         W=layer0_W,
                                         b=layer0_b)
                                        
    layer1 = ConvPoolLayer.ConvPoolLayer(rng,
                                         layer0.output,
                                         filter_shape=(nkerns[1], nkerns[0], 9, 9),
                                         image_shape=(batch_size, nkerns[0], 32, 40),
                                         W=layer1_W,
                                         b=layer1_b)
                                         
    layer2 = ConvPoolLayer.ConvPoolLayer(rng,
                                         layer1.output,
                                         filter_shape=(nkerns[2], nkerns[1], 5, 5),
                                         image_shape=(batch_size, nkerns[1], 12, 16),
                                         W=layer2_W,
                                         b=layer2_b)

                                         
    layer3 = MultiLayerPerceptron.HiddenLayer(rng, 
                                              layer2.output.flatten(2),
                                              nkerns[2] * 4 * 6, 
                                              600,
                                              W=layer3_W,
                                              b=layer3_b,
                                              activation=T.tanh)
                                              
                                      
    layer5 = LogisticLayer.LogisticLayer(layer3.output, 600, 6, W=layer5_W, b=layer5_b)
                                          
    predict_model = theano.function(inputs=[index],
                                    outputs=[layer5.errors(y)],
                                    givens={x:test_set_x[index * batch_size: (index+1) * batch_size],
                                            y:test_set_y[index * batch_size: (index+1) * batch_size]})
                                            
    prediction_losses = [predict_model(i) for i in xrange(n_test_batches)]
    this_prediction_loss = np.mean(prediction_losses)
    
    print this_prediction_loss
    
    

if __name__ == '__main__':
    predict_cnn()