#ifndef TOP_H
#define TOP_H

#include <ap_int.h>

const unsigned ARRAY_SIZE = 32;

typedef ap_int<32> Word;

// Set init = 1 to transfer data in
// Set end = 1 to transfer data out
// Otherwise the accelerator works using internal data
// Each invocation doubles each element
void top(
  Word in[ARRAY_SIZE],
  Word out[ARRAY_SIZE],
  unsigned in_size,
  unsigned mode,
  unsigned out_size
);

#endif
