import sys
import os
import time
import types

import numpy as np
import FixedPoint

kmax = 1.0
kmin = 0.001
hmax = 8.0
hmin = 0.01

# e and x are arrays
def rel_err(e,x):
    return np.percentile(np.abs(e/x), 80)

def kh_quantize(k, h):
    k_fix = FixedPoint.FixedPoint(16,15)
    h_fix = FixedPoint.FixedPoint(16,12)

    for n in range(len(k)):
        while abs(k[n]) < kmax/2 and abs(h[n]) < hmax/2:
            k[n] *= 2
            h[n] *= 2

        # k too large
        while abs(k[n]) > kmax and abs(h[n]) > 2*hmin:
            k[n] /= 2
            h[n] /= 2

        # h too large
        while abs(h[n]) > hmax and abs(k[n]) > 2*kmin:
            k[n] /= 2
            h[n] /= 2

        # k too small
        #while (abs(k[n]) < kmin and abs(h[n]) < hmax/2):
        #    k[n] *= 2
        #    h[n] *= 2

        # h too small
        #while (abs(h[n]) < hmin and abs(k[n]) < kmax/2):
        #    k[n] *= 2
        #    h[n] *= 2

    ki = k_fix.convert(k)
    hi = h_fix.convert(h)

    return k, h, ki, hi

if __name__ == "__main__":

    print("Loading the trained parameters...")
    params_dir = os.environ['CRAFT_BNN_ROOT'] + '/params/'
    old_file = 'weights_6L_bsf.npz'
    new_file = 'cifar10_6L_sf_nbkh.npz'

    # Load parameters
    with np.load(params_dir + old_file) as f:
        params = [f['arr_%d' % i] for i in range(len(f.files))]

    print "Old params length =", len(params)

    # layer param names
    # W, b, beta, gamma, mean, inv
    param_names = ['W', 'b', 'beta', 'gamma', 'mean', 'inv']
    param_new = []
    n_problems = 0

    for i in range(len(params)):
        off = i % len(param_names)
        name = param_names[off]

        if name == 'W':
            param_new += [params[i]]
        elif name == 'beta':
            b = params[i]
            g = params[i+1]
            u = params[i+2]
            s = params[i+3]

            k = np.float32(s*g)
            h = np.float32(b - u*s*g)
            k, h, ki, hi = kh_quantize(k, h)

            kerr = abs(k-ki)
            herr = abs(h-hi)

            for n in range(len(k)):
                if kerr[n]/k[n] > 0.01 or herr[n]/h[n] > 0.01:
                    print "i = ", i, ", n = ", n, ": ", kerr[n]/k[n], ",", herr[n]/h[n]
                    print "            k:", k[n],"->",ki[n]
                    print "            h:", h[n],"->",hi[n]
                    n_problems += 1

            #print "rel_error:  k =", rel_err(kerr, k), ", h = ", rel_err(herr, h)

            #print "k =", k[0:5]
            #print "h =", h[0:5]

            param_new += [k]
            param_new += [h]

    print ""
    print "n_problems =", n_problems
    np.savez(params_dir + new_file, *[param_new[x] for x in range(len(param_new))])
    print "New params length =", len(param_new)

    with np.load(params_dir + new_file) as f:
        params = [f['arr_%d' % i] for i in range(len(f.files))]
    print "Check params length =", len(params)
