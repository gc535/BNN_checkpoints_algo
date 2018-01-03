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
    test_instances = 10000
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

    X = test_set.X[0:test_instances]
    y = test_set.y[0:test_instances]

    #print('Quantizing test inputs')
    #X = FixedPoint.FixedPoint(32,31).convert(X)

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

    cnn = hardware_net.Conv2DLayer(
            cnn,
            H=H,
            num_filters=128,
            filter_size=(3, 3),
            pad=1,
            nobias=no_bias,
            nonlinearity=None)

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

    cnn = hardware_net.Conv2DLayer(
            cnn,
            H=H,
            num_filters=256,
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

    cnn = hardware_net.Conv2DLayer(
            cnn,
            H=H,
            num_filters=256,
            filter_size=(3, 3),
            pad=1,
            nobias=no_bias,
            nonlinearity=None)

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

    cnn = hardware_net.Conv2DLayer(
            cnn,
            H=H,
            num_filters=512,
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

    cnn = hardware_net.Conv2DLayer(
            cnn,
            H=H,
            num_filters=512,
            filter_size=(3, 3),
            pad=1,
            nobias=no_bias,
            nonlinearity=None)

    #--------------------------------------------
    # MP
    #--------------------------------------------
    cnn = lasagne.layers.MaxPool2DLayer(cnn, pool_size=(2, 2))

    #--------------------------------------------
    # BNNL-FC-BNNL-FC-BNNL-FC-BNNL
    #--------------------------------------------
    cnn = hardware_net.BatchNormLayer(
            cnn,
            epsilon=epsilon,
            alpha=alpha)

    cnn = lasagne.layers.NonlinearityLayer(
            cnn,
            nonlinearity=activation)

    cnn = hardware_net.DenseLayer(
                cnn,
                H=H,
                nobias=no_bias,
                nonlinearity=None,
                num_units=1024)

    cnn = hardware_net.BatchNormLayer(
            cnn,
            epsilon=epsilon,
            alpha=alpha)

    cnn = lasagne.layers.NonlinearityLayer(
            cnn,
            nonlinearity=activation)

    cnn = hardware_net.DenseLayer(
                cnn,
                H=H,
                nobias=no_bias,
                nonlinearity=None,
                num_units=1024)

    cnn = hardware_net.BatchNormLayer(
            cnn,
            epsilon=epsilon,
            alpha=alpha)

    cnn = lasagne.layers.NonlinearityLayer(
            cnn,
            nonlinearity=activation)

    cnn = hardware_net.DenseLayer(
                cnn,
                H=H,
                nobias=no_bias,
                nonlinearity=None,
                num_units=10)

    cnn = hardware_net.BatchNormLayer(
            cnn,
            epsilon=epsilon,
            alpha=alpha)

    #--------------------------------------------
    # CNN end
    #--------------------------------------------

    test_output = lasagne.layers.get_output(cnn, deterministic=True)
    test_loss = T.mean(T.sqr(T.maximum(0.,1.-target*test_output)))
    test_err = T.mean(T.neq(T.argmax(test_output, axis=1), T.argmax(target, axis=1)),dtype=theano.config.floatX)

    # Compile a second function computing the validation accuracy:
    val_fn = theano.function([input, target], test_err)

    print("Loading the trained parameters and binarizing the weights...")

    # Load parameters
    with np.load(params_dir + '/cifar10_6L_sf_nbkh.npz') as f:
        param_values = [f['arr_%d' % i] for i in range(len(f.files))]

    # params    : W, b, beta, gamma, u, v
    #print "orig param length:", len(param_values)
    #param_new = []
    #for i in range(len(param_values)):
    #    r = i % 6
    #    # W is kept
    #    if r == 0:
    #        param_new += [param_values[i]]      # W
    #        param_new += [param_values[i+1]]    # b
    #    elif r == 2:
    #        beta = param_values[i]
    #        gamma = param_values[i+1]
    #        mean = param_values[i+2]
    #        inv_std = param_values[i+3]
    #        # k
    #        param_new += [np.float32(inv_std*gamma)]
    #        # h
    #        param_new += [np.float32(beta - mean*inv_std*gamma)]

    param_new = param_values
    print "param length:", len(param_new)
    lasagne.layers.set_all_param_values(cnn, param_new)

    k_fix = FixedPoint.FixedPoint(16,15)
    h_fix = FixedPoint.FixedPoint(16,12)

    # Binarize the weights
    params = lasagne.layers.get_all_params(cnn)
    for param in params:
        if param.name == "W":
            param.set_value(hardware_net.SignNumpy(param.get_value()))
        elif param.name == "b":
            param.set_value(param.get_value())
        elif param.name == "k":
            param.set_value(k_fix.convert(param.get_value()))
        elif param.name == "h":
            param.set_value(h_fix.convert(param.get_value()))
        else:
            print "Incorrect param name", param.name
            exit(-1)
        #print param.name, param.get_value().flatten()[0:6]

    print('Running...')

    start_time = time.time()

    test_error = val_fn(X,y)*100.
    print("test_error = " + str(test_error) + "%")

    run_time = time.time() - start_time
    print("run_time = "+str(run_time)+"s")

