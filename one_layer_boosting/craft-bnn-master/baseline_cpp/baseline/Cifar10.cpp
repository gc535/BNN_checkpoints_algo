#include "Cifar10.h"

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
) {
  //#pragma HLS ARRAY_PARTITION variable= cyclic factor=32*32
  #pragma HLS ARRAY_PARTITION variable=binary_buffer.data cyclic factor=64
  #pragma HLS ARRAY_PARTITION variable=conv_buffer.data cyclic factor=64

  cl2.get_output(binary_buffer, conv_buffer);
  nl2.get_output(conv_buffer, binary_buffer);
   cl3.get_output(binary_buffer, conv_buffer);
   nl3.get_output(conv_buffer, binary_buffer);
  // cl4.get_output(binary_buffer, conv_buffer);
  // nl4.get_output(conv_buffer, binary_buffer);
  // cl5.get_output(binary_buffer, conv_buffer);
  // nl5.get_output(conv_buffer, binary_buffer);
  // cl6.get_output(binary_buffer, conv_buffer);
  // nl6.get_output(conv_buffer, binary_buffer);
}
