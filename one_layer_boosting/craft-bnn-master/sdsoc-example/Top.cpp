#include "Top.h"

#pragma SDS data copy(in[0:in_size])
#pragma SDS data copy(out[0:out_size])
void top(
  Word in[ARRAY_SIZE],
  Word out[ARRAY_SIZE],
  unsigned in_size,
  unsigned mode,
  unsigned out_size
) {
  // local storage
  static Word mem[2][ARRAY_SIZE];

  ap_uint<1> src_i = mode;
  ap_uint<1> dst_i = !mode;

  for (unsigned i = 0; i < in_size; ++i) {
    #pragma HLS PIPELINE
    mem[src_i][i] = in[i];
  }

  LOOP_MAIN:
  for (unsigned i = 0; i < ARRAY_SIZE; ++i) {
    mem[dst_i][i] = 2*mem[src_i][i];
  }

  for (unsigned i = 0; i < out_size; ++i) {
    #pragma HLS PIPELINE
    out[i] = mem[dst_i][i];
  }
}
