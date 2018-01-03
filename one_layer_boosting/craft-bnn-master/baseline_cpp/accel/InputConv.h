#ifndef ACCEL_INPUT_CONV_H
#define ACCEL_INPUT_CONV_H

#include "Typedefs.h"
#include "Accel.h"

void input_conv_layer_cpu(
    const uint64_t* w,
    const float* k,
    const float* h,
    const float* data_i,
    uint64_t* data_o,
    const unsigned M,
    const unsigned N
);

#endif
