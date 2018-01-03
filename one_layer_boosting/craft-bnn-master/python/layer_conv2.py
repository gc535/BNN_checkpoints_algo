import sys
import os
import time

import numpy as np
np.random.seed(1234) # for reproducibility?

# specifying the gpu to use
# import theano.sandbox.cuda
# theano.sandbox.cuda.use('gpu1')
import theano
import theano.tensor as T
import lasagne

import cPickle as pickle
import gzip

import hardware_net
import FixedPoint
import Printer

from pylearn2.datasets.cifar10 import CIFAR10

if __name__ == "__main__":

    # BN parameters
    # alpha is the exponential moving average factor
    alpha = .1
    epsilon = 1e-4

    # Parameters directory
    if not os.environ.has_key('CRAFT_BNN_ROOT'):
        print "CRAFT_BNN_ROOT not set!"
        exit(-1)
    top_dir = os.environ['CRAFT_BNN_ROOT']
    params_dir = top_dir + '/params'

    # BinaryOut
    activation = hardware_net.SignTheano
    print("activation = sign(x)")

    no_bias = True
    print("no_bias = " + str(no_bias))

    # BinaryConnect
    H = 1.
    print('Loading CIFAR-10 dataset...')

    test_set = CIFAR10(which_set="test")
    print("Test set size = "+str(len(test_set.X)))
    test_instances = 1
    print("Using instances 0 .. "+str(test_instances))

    # bc01 format
    # Inputs in the range [-1,+1]
    test_set.X = np.reshape(np.subtract(np.multiply(2./255.,test_set.X),1.),(-1,3,32,32))
    # flatten targets
    test_set.y = np.hstack(test_set.y)
    # Onehot the targets
    test_set.y = np.float32(np.eye(10)[test_set.y])
    # for hinge loss
    test_set.y = 2* test_set.y - 1.

    print('Quantizing the input...')
    X = test_set.X[0:test_instances]
    X = FixedPoint.FixedPoint(32,31).convert(X)
    y = test_set.y[0:test_instances]

    #X = np.float32( np.zeros(X.shape) )
    #X[0,0] = 0.1
    #X[0,1] = 0.4
    #X[0,2] = -0.9

    print('Building the CNN...')

    # Prepare Theano variables for inputs and targets
    input = T.tensor4('inputs')
    target = T.matrix('targets')
    LR = T.scalar('LR', dtype=theano.config.floatX)

    #--------------------------------------------
    # CNN start
    #--------------------------------------------
    cnn = lasagne.layers.InputLayer(
            shape=(None, 3, 32, 32),
            input_var=input)

    #--------------------------------------------
    cnn = hardware_net.Conv2DLayer(
            cnn,
            H=H,
            num_filters=128,
            filter_size=(3, 3),
            pad=1,
            nobias=no_bias,
            nonlinearity=None)

    cnn = hardware_net.BatchNormLayer(
            cnn,
            epsilon=epsilon,
            alpha=alpha)

    cnn = lasagne.layers.NonlinearityLayer(
            cnn,
            nonlinearity=activation)

    layer_input = lasagne.layers.get_output(cnn, deterministic=True)

    cnn = hardware_net.Conv2DLayer(
            cnn,
            H=H,
            num_filters=128,
            filter_size=(3, 3),
            pad=1,
            nobias=no_bias,
            nonlinearity=None)

    conv_output = lasagne.layers.get_output(cnn, deterministic=True)

    #--------------------------------------------
    # MP
    #--------------------------------------------
    cnn = lasagne.layers.MaxPool2DLayer(cnn, pool_size=(2, 2))

    #--------------------------------------------
    # BNNL-Conv-BNNL-Conv
    #--------------------------------------------
    cnn = hardware_net.BatchNormLayer(
            cnn,
            epsilon=epsilon,
            alpha=alpha)

    cnn = lasagne.layers.NonlinearityLayer(
            cnn,
            nonlinearity=activation)

    #--------------------------------------------
    # CNN end
    #--------------------------------------------

    layer_output = lasagne.layers.get_output(cnn, deterministic=True)

    # Compile a function to produce the layer output
    input_fn = theano.function([input], layer_input)
    conv_fn = theano.function([input], conv_output)
    cnn_fn = theano.function([input], layer_output)

    print("Loading the trained parameters and binarizing the weights...")

    # Load parameters
    with np.load(params_dir + '/cifar10_parameters_nb.npz') as f:
        param_values = [f['arr_%d' % i] for i in range(6)]

    lasagne.layers.set_all_param_values(cnn, param_values)

    k_fix = FixedPoint.FixedPoint(16,15)
    h_fix = FixedPoint.FixedPoint(16,12)

    # Binarize the weights
    params = lasagne.layers.get_all_params(cnn)
    l = 1
    lout = 2
    for param in params:
        if param.name == "W":
            param.set_value(hardware_net.SignNumpy(param.get_value()))
            if l == lout:
                print "kernel 0 for layer", l
                Printer.print_3d(param.get_value()[0,:,:,:], 3, 3, 3, 'b')
        elif param.name == "k":
            param.set_value(k_fix.convert(param.get_value()))
            if l == lout:
                print "k =", param.get_value()[0:4]
        elif param.name == "h":
            param.set_value(h_fix.convert(param.get_value()))
            if l == lout:
                print "h =", param.get_value()[0:4]
            l = l + 1
        else:
            print "Incorrect param name", param.name
            exit(-1)

    print('Running...')

    start_time = time.time()

    input_ = input_fn(X)
    output = cnn_fn(X)
    conv_out = conv_fn(X)
    print "input shape=", input_.shape
    Printer.print_2d(input_[0,0,:,:], 8, 8, 'b')
    print " ##"
    Printer.print_2d(input_[0,1,:,:], 8, 8, 'b')

    print "\nconv shape=", conv_out.shape
    Printer.print_2d(conv_out[0,0,:,:], 8, 8, 'i')

    print "\noutput shape=", output.shape
    Printer.print_2d(output[0,0,:,:], 8, 8, 'b')
    print " ##"
    Printer.print_2d(output[0,1,:,:], 8, 8, 'b')
    print " ##"
    Printer.print_2d(output[0,2,:,:], 8, 8, 'b')

    #np.savez("py_conv2_maps.npz", output);

    run_time = time.time() - start_time
    print("run_time = "+str(run_time)+"s")

