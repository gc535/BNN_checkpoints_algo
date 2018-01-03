//------------------------------------------------------------------------
// Definitions for the Cifar10 BNN architecture
//------------------------------------------------------------------------
#ifndef CIFAR10_H
#define CIFAR10_H

#include "Common.h"
#include "Layers.h"

const char params_file[]  = "/params/cifar10_parameters_nb.zip";

const unsigned K = 3;

typedef InputConvLayerSW<3,128,32,K>ConvLayer1;
//typedef InputConvLayer<3,128,32,K>  ConvLayer1;
typedef Conv2Layer<128,128,32,K>     ConvLayer2;
typedef Max2NormLayer<128,32>       NormLayer2;
typedef Conv2Layer<128,256,16,K>     ConvLayer3;
typedef BatchNormLayer<256,16>      NormLayer3;
typedef ConvLayer<256,256,16,K>     ConvLayer4;
typedef Max2NormLayer<256,16>       NormLayer4;
typedef ConvLayer<256,512,8,K>      ConvLayer5;
typedef BatchNormLayer<512,8>       NormLayer5;
typedef ConvLayer<512,512,8,K>      ConvLayer6;
typedef Max2NormLayer<512,8>        NormLayer6;
typedef DenseLayer<512*16,1024>     DenseLayer1;
typedef BatchNormLayer<1024,1>      NormLayer7;
typedef DenseLayer<1024,1024>       DenseLayer2;
typedef BatchNormLayer<1024,1>      NormLayer8;
typedef DenseLayer<1024,10>         DenseLayer3;

//------------------------------------------------------------------------
// HLS top
//------------------------------------------------------------------------
void top(
    SArray<Bit,        128*32*32>     &binary_buffer,
    SArray<ConvOutput, 128*32*32>     &conv_buffer,
    const ConvLayer2 &cl2,
    const NormLayer2 &nl2,
     const ConvLayer3 &cl3,
     const NormLayer3 &nl3
    // const ConvLayer4 &cl4,
    // const NormLayer4 &nl4,
    // const ConvLayer5 &cl5,
    // const NormLayer5 &nl5,
    // const ConvLayer6 &cl6,
    // const NormLayer6 &nl6
);

#endif
