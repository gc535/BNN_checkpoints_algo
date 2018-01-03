#include <cstddef>
#include <hls_video.h>

#include "Accel.h"
#include "AccelSchedule.h"
#include "AccelTest.h"
#include "Conv.h"

const unsigned N = 2;

//------------------------------------------------------------------------
// Helper test function for the accelerator, random data
//------------------------------------------------------------------------
void test_conv_layer_random(
    const unsigned S,
    Word* wt,
    Word* kh
) {
  const unsigned M = CONVOLVERS*PIX_PER_PHASE / (S*S);

  // Generate the input data
  assert (M*S*S <= DMEM_WORDS*WORD_SIZE);
  Word* data_i = (Word*) MEM_ALLOC( DMEM_WORDS * sizeof(Word) );
  for (unsigned m = 0; m < M; ++m) {
    for (unsigned r = 0; r < S; ++r) {
      for (unsigned c = 0; c < S; ++c) {
        set_bit(data_i, m*S*S+r*S+c, simple_hash(m*S*S+r*S+c));
  }  }  }

  assert (N*S*S <= DMEM_O_WORDS*WORD_SIZE);
  Word* data_o = (Word*) MEM_ALLOC( DMEM_O_WORDS * sizeof(Word) );

  DB(2,
    printf ("*data*:\n");
    print_bits3d(data_i, 0, 2, S, 8,S);
    printf ("*params*:\n");
    print_bits3d(wt, 0, 2, K, K,K);
  );

  // Compute conv reference
  int conv_ref[S*S];
  bin_conv((uint64_t*)wt, (uint64_t*)data_i, conv_ref, M, 1, S);

  // Compute bin reference
  Word bin_ref[S*S];
  Word khword = kh[0];
  NormComp nc;  nc(15,0) = khword(15,0);    // nc = -h/k
  float k = 1.0;
  float h = -nc.to_int();
  bin_conv_binarize(
      &k, &h,
      conv_ref,
      (uint64_t*)bin_ref,
      1, S
  );

  test_conv_layer(
      wt, kh, data_i, data_o, bin_ref,
      M, 1, S
    );

  MEM_FREE( data_i );
  MEM_FREE( data_o );
}

//------------------------------------------------------------------------
// Main
//------------------------------------------------------------------------
int main() {
  Word* wt = new Word[WT_WORDS];
  Word* kh = new Word[KH_WORDS];

  // initialize the kernel weights
  for (unsigned m = 0; m < WT_WORDS; ++m) {
    for (unsigned i = 0; i < WORD_SIZE; ++i)
      set_bit(wt, m*WORD_SIZE+i, simple_hash(m*WORD_SIZE+i));
  }
  // initialize the batch-norm params
  for (unsigned n = 0; n < N; ++n) {
    NormComp nc = 10 + 10*n;

    int off = n % KH_PER_WORD;

    Word w = kh[n/KH_PER_WORD];
    w((off+1)*16-1, off*16) = nc(15,0);
    kh[n/KH_PER_WORD] = w;
  }

  test_conv_layer_random( 8, wt, kh);
  test_conv_layer_random(16, wt, kh);
  test_conv_layer_random(32, wt, kh);

  delete[] wt;
  delete[] kh;

  printf ("Tests passed!\n");
  return 0;
}
