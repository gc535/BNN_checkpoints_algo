//------------------------------------------------------------------------
// Utility functions for param quantization
//------------------------------------------------------------------------
#ifndef QUANTIZE_PARAMS_H
#define QUANTIZE_PARAMS_H

#include <vector>
#include <string>
#include <iostream>
#include <minizip/zip.h>

#include "ParamIO.h"
#include "ZipIO.h"
#include "BitVector.h"

//------------------------------------------------------------------------
// Turns an array [p] of floats with length [len] into an array of fixed
// point values.
// FixedT is the ap_fixed type to do the quantization
// StoreT is the C-type with equivalent bitwidth
//------------------------------------------------------------------------
template<typename FixedT, typename StoreT>
std::vector<StoreT>
quantize_array(float* p, unsigned len) {
  std::vector<StoreT> buf(len, 0);

  for (int i = 0; i < len; ++i) {
    FixedT fi = p[i];
    if (i < 6)
      DB(3,  std::cout << "  " << p[i] << " -> " << fi << "\n" );
    buf[i] = fi.range(sizeof(StoreT)*8-1,0);
  }

  return buf;
}

//------------------------------------------------------------------------
// Calls the two functions above
//------------------------------------------------------------------------
template<typename FixedT, typename StoreT>
void quantize_array_and_write_to_zip(
    zipFile zf,
    std::string fname,
    const Params &params,
    unsigned idx      // index to params
    )
{
  float* p = (float*)( params.array_data(idx) );
  unsigned len = params.array_size(idx)/4;
  
  DB_PRINT(3, "Quantizing %10s, %u bytes original\n", fname.c_str(), len*4);
  auto buf = quantize_array<FixedT, StoreT> (p, len);
  write_buffer_to_zip(zf, fname, (void*)buf.data(), len*sizeof(StoreT));
}

#endif
