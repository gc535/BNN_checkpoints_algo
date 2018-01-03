//------------------------------------------------------------------------
// Applies quantization to cifar10_parameters_nb.zip
//------------------------------------------------------------------------
#include <ap_fixed.h>
#include <vector>
#include <string>
#include <iostream>
#include <minizip/zip.h>

#include "QuantizeParams.h"
#include "ParamIO.h"
#include "Common.h"
#include "BitVector.h"

const char infile[]  = "/params/cifar10_parameters_nb.zip";
const char outfile[] = "/params/cifar10_parameters_nb_fixed.zip";
const unsigned N = 3;

typedef ap_fixed<16,1, AP_RND> KFixed;
typedef ap_fixed<16,4, AP_RND> HFixed;

//------------------------------------------------------------------------
int main() {
  // Read all the params
  std::string full_infile = get_root_dir() + infile;
  Params params(infile);
  
  std::string full_outfile = get_root_dir() + outfile;
  zipFile zf = zipOpen(full_outfile.c_str(), 0);

  // The params are organized 'W', 'k', 'h' x9 for 27 arrays total
  assert(params.num_arrays() == 27);

  printf ("Writing zip archive %s\n", outfile);

  // Go through the 9 layers and quantize the 4 arrays in each layer
  for (int n = 0; n < 9; ++n) {

    //-------------------------------------
    // Binary quantization for 'W'
    {
      unsigned idx = N*n;
      float* p = (float*)( params.array_data(idx) );
      unsigned len = params.array_size(idx)/4;  // convert from bytes to floats

      BitVector bv(len);

      // Binarize, negatives are 1 and positives are 0
      for (int i = 0; i < len; ++i)
        if (p[i] < 0)
          bv.set(i);

      std::string fname = "W_" + std::to_string(n);
      write_buffer_to_zip(zf, fname, (void*)bv.data(), bv.bytesize());
    }
    
    //-------------------------------------
    // 16-bit quantization for 'k'
    {
      unsigned idx = N*n+1;
      std::string fname = "K_" + std::to_string(n);
      quantize_array_and_write_to_zip<KFixed, uint16_t> (zf, fname, params, idx);
    }

    //-------------------------------------
    // 16-bit quantization for 'h'
    {
      unsigned idx = N*n+2;
      std::string fname = "H_" + std::to_string(n);
      quantize_array_and_write_to_zip<HFixed, uint16_t> (zf, fname, params, idx);
    }
  }

  int err = zipClose(zf, NULL);
  assert(err == ZIP_OK);

  return 0;
}
