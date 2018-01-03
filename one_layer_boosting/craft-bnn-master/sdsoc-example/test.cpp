#include <assert.h>
#include "Top.h"
#include "Timer.h"

const unsigned DATA_SIZE = 4*ARRAY_SIZE;

int main() {
  printf ("Begin sdsoc-example test\n");
  Word* data_i = new Word[DATA_SIZE];
  Word* data_o = new Word[DATA_SIZE];

  printf ("Generating input data\n");
  for (unsigned i = 0; i < DATA_SIZE; ++i)
    data_i[i] = i % 256;

  Word in[ARRAY_SIZE];
  Word out[ARRAY_SIZE];

  Timer t_accel("accel");

  printf ("Begin test loop\n");
  for (unsigned n = 0; n < DATA_SIZE; n+=ARRAY_SIZE) {
    // copy input data
    printf ("%2d: Copying input data\n", n/ARRAY_SIZE);
    for (unsigned i = 0; i < ARRAY_SIZE; ++i) {
      in[i] = data_i[n+i];
    }

    printf ("    Invoking accelerator\n");
    t_accel.start();
    top( in, out, ARRAY_SIZE, 0, 0 );  // data input, double
    top( in, out, 0, 1, ARRAY_SIZE );  // data output, double
    t_accel.stop();

    // copy output data
    printf ("    Reading output data\n");
    for (unsigned i = 0; i < ARRAY_SIZE; ++i) {
      data_o[n+i] = out[i];
    }
  }

  printf ("Checking results\n");
  for (unsigned i = 0; i < DATA_SIZE; ++i) {
    const Word y = 4*data_i[i];
    if (data_o[i] != y) {
      printf ("i=%d, %d != %d\n", i, data_o[i].to_int(), y.to_int());
      exit(-1);
    }
  }

  printf ("Tests passed!\n");
  delete[] data_i;
  delete[] data_o;
}
