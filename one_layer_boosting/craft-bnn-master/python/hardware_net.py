import time
import numpy as np

import theano
import theano.tensor as T

import lasagne


# We will not use any of these layers in training, so we can cut a lot
# of corners

# This sends a distribution between -1 and 1 to 0 and 1, scaling by 0.5 and
# adding +0.5 y-offset
def hard_sigmoid(x):
    return T.clip((x+1.)/2., 0,1)

# Weight binarization function
def SignNumpy(x):
    return np.float32(2.*np.greater_equal(x,0)-1.)

# Activation binarization function
def SignTheano(x):
    return T.cast(2.*T.ge(x,0)-1., theano.config.floatX)

# The weights' binarization function, 
# taken directly from the BinaryConnect github repository and simplified
# (which was made available by his authors)
def binarization(W, H, binary=True):

    if not binary:
        Wb = W
    else:
        # [-1,1] -> [0,1]
        Wb = hard_sigmoid(W/H)
        Wb = T.round(Wb)
        
        # 0 or 1 -> -1 or 1
        Wb = T.cast(T.switch(Wb,H,-H), theano.config.floatX)
    
    return Wb

#------------------------------------------------------------------------
# This class extends the Lasagne Conv2DLayer to support BinaryConnect
#------------------------------------------------------------------------
class DenseLayer(lasagne.layers.DenseLayer):

    def __init__(self, incoming, num_units, H=1., nobias=False, **kwargs):
        self.binary = True
        self.H = H

        if H == "Glorot":
            num_inputs = int(np.prod(incoming.output_shape[1:]))
            self.H = np.float32(np.sqrt(1.5/ (num_inputs + num_units)))

        if nobias:
            super(DenseLayer, self).__init__(incoming, num_units, \
                  W=lasagne.init.Uniform((-self.H,self.H)), b=None, **kwargs)
        else:
            super(DenseLayer, self).__init__(incoming, num_units, \
                  W=lasagne.init.Uniform((-self.H,self.H)), **kwargs)
        # add the binary tag to weights            
        self.params[self.W]=set(['binary'])
        
    def get_output_for(self, input, deterministic=False, **kwargs):
        #self.Wb = binarization(self.W, self.H, self.binary)
        #Wr = self.W
        #self.W = self.Wb
            
        rvalue = super(DenseLayer, self).get_output_for(input, **kwargs)
        #self.W = Wr
        return rvalue

#------------------------------------------------------------------------
# This class extends the Lasagne Conv2DLayer to support BinaryConnect
#------------------------------------------------------------------------
class Conv2DLayer(lasagne.layers.Conv2DLayer):
    
    def __init__(self, incoming, num_filters, filter_size, H=1., nobias=False, **kwargs):
        self.binary = True
        self.H = H

        if H == "Glorot":
            num_inputs = int(np.prod(filter_size)*incoming.output_shape[1])
            num_units = int(np.prod(filter_size)*num_filters)
            # theoretically, I should divide num_units by the pool_shape
            self.H = np.float32(np.sqrt(1.5 / (num_inputs + num_units)))
    
        if nobias:
            super(Conv2DLayer, self).__init__(incoming, num_filters, filter_size, \
                  W=lasagne.init.Uniform((-self.H,self.H)), b=None, **kwargs)
        else:
            super(Conv2DLayer, self).__init__(incoming, num_filters, filter_size, \
                  W=lasagne.init.Uniform((-self.H,self.H)), **kwargs)
        # add the binary tag to weights            
        self.params[self.W]=set(['binary'])
    
    def convolve(self, input, deterministic=False, **kwargs):
        rvalue = super(Conv2DLayer, self).convolve(input, **kwargs)
        return rvalue

#------------------------------------------------------------------------
# This class extends the Lasagne BatchNorm layer to reduce the parameters
#------------------------------------------------------------------------
class BatchNormLayer(lasagne.layers.Layer):

    def __init__(self, incoming, axes='auto', epsilon=1e-4, alpha=0.1,
                 **kwargs):

        # This part is copied from Lasagne
        # -----
        super(BatchNormLayer,self).__init__(incoming, **kwargs)
        self.epsilon = epsilon
        self.alpha = alpha

        if axes == 'auto':
            axes = (0,) + tuple(range(2, len(self.input_shape)))
        elif isistance(axes, int):
            axes = (axes,)
        self.axes = axes

        shape = [size for axis, size in enumerate(self.input_shape)
                 if axis not in self.axes]
        if any(size is None for size in shape):
            raise ValueError("BatchNormLayer needs specified input sizes for "
                             "all axes not normalized over.")
        # -----

        # Our code
        ini = np.float32(np.zeros(shape))
        self.k = self.add_param(ini, shape, name='k',
                                trainable=False, regularizable=False)
        self.h = self.add_param(ini, shape, name='h',
                                trainable=False, regularizable=False)

    def get_output_for(self, input, **kwargs):
        k = self.k
        h = self.h

        # This part is copied from Lasagne
        # -----
        # prepare dimshuffle pattern inserting broadcastable axes as needed
        param_axes = iter(range(input.ndim - len(self.axes)))
        pattern = ['x' if input_axis in self.axes
                   else next(param_axes)
                   for input_axis in range(input.ndim)]

        # apply dimshuffle pattern to all parameters
        k = 0 if self.k is None else self.k.dimshuffle(pattern)
        h = 0 if self.h is None else self.h.dimshuffle(pattern)
        # -----

        normalized = input*k + h
        return normalized
        







