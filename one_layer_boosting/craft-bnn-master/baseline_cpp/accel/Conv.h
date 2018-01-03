#ifndef ACCEL_CONV_H
#define ACCEL_CONV_H

#include "Debug.h"
#include "Typedefs.h"
#include "Accel.h"

// binary convolution
// [w] and [in] are arrays of packed binary values
// [conv_out] is an array of integers
void bin_conv(
    const uint64_t* w, const uint64_t* in,
    int* conv_out,
    unsigned M, unsigned N,
    unsigned S
);

// batch norm binarization
void bin_conv_binarize(
    const float* k,
    const float* h,
    const int* conv_res,
    uint64_t* bin_out,
    unsigned N,
    unsigned S
);

// batch norm binarization with pooling
void bin_conv_binarize_pool(
    const float* k,
    const float* h,
    const int* conv_res,
    uint64_t* bin_out,
    unsigned N,
    unsigned S
);

void bin_conv_layer_cpu(
    const uint64_t* w,
    const float* k,
    const float* h,
    const uint64_t* data_i,
    uint64_t* data_o,
    const unsigned M,
    const unsigned N,
    const unsigned width_mode,
    const unsigned max_pool
);

#endif
