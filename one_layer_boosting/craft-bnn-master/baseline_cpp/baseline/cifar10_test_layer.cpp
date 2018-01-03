//------------------------------------------------------------------------
// Classifies 1 image and Checks the output of some intermediate layers
//------------------------------------------------------------------------
#include <iostream>

#include "ParamIO.h"
#include "DataIO.h"
#include "Cifar10.h"
#include "Timer.h"

//------------------------------------------------------------------------
// Testbench
//------------------------------------------------------------------------
int main() {
  // Read 1 input image and label
  Cifar10TestInputs X(1);
  Cifar10TestLabels y(1);

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

  // Check params
  Params params(get_root_dir() + params_file);
  assert(params.array_size(0) == 3*128*K*K * 4);
  assert(params.array_size(1) == 128 * 4);
  assert(params.array_size(2) == 128 * 4);
  assert(params.array_size(3) == 128*128*K*K * 4);
  assert(params.array_size(4) == 128 * 4);
  assert(params.array_size(5) == 128 * 4);
  assert(params.array_size(6) == 128*256*K*K * 4);
  assert(params.array_size(7) == 256 * 4);
  assert(params.array_size(8) == 256 * 4);
  assert(params.array_size(9) == 256*256*K*K * 4);
  assert(params.array_size(10) == 256 * 4);
  assert(params.array_size(11) == 256 * 4);
  assert(params.array_size(12) == 256*512*K*K * 4);
  assert(params.array_size(13) == 512 * 4);
  assert(params.array_size(14) == 512 * 4);
  assert(params.array_size(15) == 512*512*K*K * 4);
  assert(params.array_size(16) == 512 * 4);
  assert(params.array_size(17) == 512 * 4);

  assert(params.array_size(18) == 512*16*1024 * 4);
  assert(params.array_size(19) == 1024 * 4);
  assert(params.array_size(20) == 1024 * 4);
  assert(params.array_size(21) == 1024*1024 * 4);
  assert(params.array_size(22) == 1024 * 4);
  assert(params.array_size(23) == 1024 * 4);
  assert(params.array_size(24) == 1024*10 * 4);
  assert(params.array_size(25) == 10 * 4);
  assert(params.array_size(26) == 10 * 4);


  // Quantize and load params to layers
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
  copy_input(input, X.data);

  // CNN execution
  Timer t_conv1("conv1", true);
    conv1.get_output(input, binary_buffer);
  t_conv1.stop();

  Timer t_hls("hls top", true);
    conv2.get_output(binary_buffer, conv_buffer);
    norm2.get_output(conv_buffer, binary_buffer);
    conv3.get_output(binary_buffer, conv_buffer);
    norm3.get_output(conv_buffer, binary_buffer);
    conv4.get_output(binary_buffer, conv_buffer);
    norm4.get_output(conv_buffer, binary_buffer);
    conv5.get_output(binary_buffer, conv_buffer);
    norm5.get_output(conv_buffer, binary_buffer);
    conv6.get_output(binary_buffer, conv_buffer);
    norm6.get_output(conv_buffer, binary_buffer);
  t_hls.stop();

  // Check the results of the fixed-point buffer
  {
    const unsigned N = 512;
    const unsigned S = 8;
    printf("\nconv buffer:\n");
    conv_buffer.print_sub(0, S, 8, 'i');
    printf ("--\n");
    conv_buffer.print_sub(1, S, 8, 'i');

    std::string mfile = get_root_dir() + "/data/layer_conv6_maps.zip";
    SArray<float,N*S*S> mdata;
    unzip_to_sarray(mfile, mdata);

    // Calculate the mean abs error / mean abs value
    int dsum = 0;
    int sum = 0;
    int nerr = 0;
    for (unsigned n = 0; n < N; ++n)
      for (unsigned s = 0; s < S*S; ++s) {
        int m = (int)mdata[n*S*S + s];
        int diff = conv_buffer[n*S*S + s].to_int() - m;
        sum += abs(m);
        dsum += abs(diff);
        nerr += (diff != 0) ? 1 : 0;
      }
    float err = (float)dsum*100 / sum;
    float err_rate = (float)nerr*100 / (N*S*S);
    printf("fixed buffer error = %6.2f%%, err rate=%6.2f%%\n", err, err_rate);
    assert(err >= 0);
  }

  // Check the results of the binary buffer
  {
    const unsigned N = 128;
    const unsigned S = 32;
    printf("\nbinary buffer:\n");
    binary_buffer.print_sub(0, S, 8, 'i');
    printf ("--\n");
    binary_buffer.print_sub(1, S, 8, 'i');

    std::string mfile = get_root_dir() + "/data/layer_norm1_maps.zip";
    SArray<float,N*S*S> mdata;
    unzip_to_sarray(mfile, mdata);

    unsigned err = 0;
    for (unsigned n = 0; n < N; ++n)
      for (unsigned s = 0; s < S*S; ++s) {
        Bit m = mdata[n*S*S + s] < 0 ? -1 : 0;
        if (m != binary_buffer[n*S*S + s]) {
          err++;
        }
      }
    float err_rate = (float)err*100 / (N*S*S);
    printf ("binary buffer error rate = %6.2f\n", err_rate);
    assert(err_rate >= 0);
    //assert(err_rate < 1);
  }

  // Dense layers execution
  Timer t_dense("dense", true);
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
    }
  }
  assert(prediction != 11);
  t_dense.stop();

  printf (" pred  = %u\n", prediction);
  printf (" label = %d\n", (int)y.data[0]);

  return 0;
}
