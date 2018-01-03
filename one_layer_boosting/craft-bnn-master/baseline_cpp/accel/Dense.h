#ifndef ACCEL_DENSE_H
#define ACCEL_DENSE_H

#include "Debug.h"
#include "Typedefs.h"
#include "Accel.h"

void dense_layer_cpu(
    const uint64_t* w,
    const float* k_data,
    const float* h_data,
    const uint64_t* data_i,
    uint64_t* data_o,
    const unsigned M,
    const unsigned N
);

int last_layer_cpu(
    const uint64_t* w,
    const float* k_data,
    const float* h_data,
    const uint64_t* in,
    const unsigned M,
    const unsigned N
);

#endif
