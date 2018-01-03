//------------------------------------------------------------------------
// Classifies a series of images
//------------------------------------------------------------------------
#include <iostream>

#include "ParamIO.h"
#include "DataIO.h"
#include "Cifar10.h"
#include "Timer.h"

//------------------------------------------------------------------------
// Testbench
//------------------------------------------------------------------------
int main(int argc, char** argv) {
  if (argc < 2) {
    printf ("Give number of images to test as 1st arg\n");
    return 0;
  }

  unsigned N = std::stoi(argv[1]);

  // Read input images and labels
  Cifar10TestInputs X(N);
  Cifar10TestLabels y(N);

  // CNN architecture
  ConvLayer1 conv1("conv1");
  ConvLayer2 conv2("conv2");
  NormLayer2 norm2("norm2");
  ConvLayer3 conv3("conv3");
  NormLayer3 norm3("norm3");
  ConvLayer4 conv4("conv4");
  NormLayer4 norm4("norm4");
  ConvLayer5 conv5("conv5");
  NormLayer5 norm5("norm5");
  ConvLayer6 conv6("conv6");
  NormLayer6 norm6("norm6");
  DenseLayer1 dense1("dense1");
  NormLayer7 norm7("norm7");
  DenseLayer2 dense2("dense2");
  NormLayer8 norm8("norm8");
  DenseLayer3 dense3("dense3");

  // Quantize and load params to layers
  Params params(get_root_dir() + params_file);
  conv1.load_weights(params.float_data(0));
  conv1.load_kh(params.float_data(1), params.float_data(2));

  conv2.load_weights(params.float_data(3));
  norm2.load_weights(params.float_data(4), params.float_data(5));
  conv3.load_weights(params.float_data(6));
  norm3.load_weights(params.float_data(7), params.float_data(8));
  conv4.load_weights(params.float_data(9));
  norm4.load_weights(params.float_data(10), params.float_data(11));
  conv5.load_weights(params.float_data(12));
  norm5.load_weights(params.float_data(13), params.float_data(14));
  conv6.load_weights(params.float_data(15));
  norm6.load_weights(params.float_data(16), params.float_data(17));

  dense1.load_weights(params.float_data(18));
  norm7.load_weights(params.float_data(19), params.float_data(20));
  dense2.load_weights(params.float_data(21));
  norm8.load_weights(params.float_data(22), params.float_data(23));
  dense3.load_weights(params.float_data(24));

  // Declare buffers
  SArray<Bit,        128*32*32> binary_buffer;
  SArray<ConvOutput, 128*32*32> conv_buffer;

  SArray<ConvLayer1::DataType, 3*32*32>   input;

  Timer t_dense("dense");
  Timer t_hls("hls top");
  Timer t_conv1("conv1");

  unsigned n_errors = 0;

  for (unsigned n = 0; n < N; ++n) {
    float* data = X.data + n*3*32*32;
    copy_input(input, data);

    t_conv1.start();
    conv1.get_output(input, binary_buffer);
    t_conv1.stop();

    // CNN execution
    t_hls.start();
    top(
        binary_buffer,
        conv_buffer,
        conv2,
        norm2,
         conv3,
         norm3
        // conv4,
        // norm4,
        // conv5,
        // norm5,
        // conv6,
        // norm6
      );
    t_hls.stop();

    // Dense layers execution
    t_dense.start();
   // conv3.get_output(binary_buffer, conv_buffer);
    //norm3.get_output(conv_buffer, binary_buffer);
    conv4.get_output(binary_buffer, conv_buffer);
    norm4.get_output(conv_buffer, binary_buffer);
    conv5.get_output(binary_buffer, conv_buffer);
    norm5.get_output(conv_buffer, binary_buffer);
    conv6.get_output(binary_buffer, conv_buffer);
    norm6.get_output(conv_buffer, binary_buffer);
    dense1.get_output(binary_buffer, conv_buffer);
    norm7.get_output(conv_buffer, binary_buffer);
    dense2.get_output(binary_buffer, conv_buffer);
    norm8.get_output(conv_buffer, binary_buffer);
    dense3.get_output(binary_buffer, conv_buffer);

    int prediction = 11;
    float maxval = -1e20;
    for (int i = 0; i < 10; ++i) {
      float k = params.float_data(25)[i];
      float h = params.float_data(26)[i];
      float val = conv_buffer[i] * k + h;
      if (val > maxval) {
        
        prediction = i;
        maxval = val;
        printf ("val=: %f maxval = %f \n", val, maxval);
      }
    }
    t_dense.stop();

    int label = y.data[n];
	
    printf ("  Pred/Label:\t%2u/%2d\t[%s]\n", prediction, label,
        ((prediction==label)?" OK ":"FAIL"));

    n_errors += (prediction!=label);
  }

  printf ("\n");
  printf ("Errors: %u\n", n_errors);
  printf ("\n");

  return 0;
}
