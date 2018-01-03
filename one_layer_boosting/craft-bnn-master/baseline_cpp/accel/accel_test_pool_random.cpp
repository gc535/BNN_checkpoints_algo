#include <cstddef>
#include <hls_video.h>

#include "Accel.h"
#include "AccelSchedule.h"
#include "AccelTest.h"
#include "Conv.h"

const unsigned N = 4;

//------------------------------------------------------------------------
// Helper test function for the accelerator, random data
//------------------------------------------------------------------------
void test_conv_pool_layer_random(
    const unsigned Si,
    Word* wt,
    Word* kh
) {
  const unsigned M = CONVOLVERS*PIX_PER_PHASE / (Si*Si);
  const unsigned So = Si / POOL_WIDTH;

  // Generate the input data
  assert (M*Si*Si <= DMEM_WORDS*WORD_SIZE);
  Word* data_i = (Word*) MEM_ALLOC( DMEM_WORDS * sizeof(Word) );
  for (unsigned m = 0; m < M; ++m) {
    for (unsigned r = 0; r < Si; ++r) {
      for (unsigned c = 0; c < Si; ++c) {
        set_bit(data_i, m*Si*Si+r*Si+c, simple_hash(m*Si*Si+r*Si+c));
  }  }  }

  assert (N*So*So <= DMEM_O_WORDS*WORD_SIZE);
  Word* data_o = (Word*) MEM_ALLOC( DMEM_O_WORDS * sizeof(Word) );

  DB(2,
    printf ("*data*:\n");
    print_bits3d(data_i, 0, 2, Si, 8,Si);
    printf ("*params*:\n");
    print_bits3d(wt, 0, 2, K, K,K);
  );

  // Compute conv reference
  int conv_ref[N*Si*Si];
  Word bin_ref[N*So*So];
  //padded_conv(data_i, wt, conv_ref, M, N, Si);
  bin_conv((uint64_t*)wt, (uint64_t*)data_i, conv_ref, M, N, Si);

  // Compute bin reference
  for (unsigned n = 0; n < N; ++n) {
    Word kh_word = kh[n/KH_PER_WORD];
    kh_word = kh_word >> (n%KH_PER_WORD)*16;
    NormComp nc;
    nc(15,0) = kh_word(15,0);

    for (unsigned r = 0; r < So; ++r) {
    for (unsigned c = 0; c < So; ++c) {
      // load from conv_ref
      unsigned idx = n*Si*Si + POOL_WIDTH*r*Si + POOL_WIDTH*c;
      Word p = conv_ref[idx];
      // max pooling
      for (unsigned pr = 0; pr < POOL_WIDTH; ++pr) {
      for (unsigned pc = 0; pc < POOL_WIDTH; ++pc) {
        Word _p = conv_ref[idx + pr*Si + pc];
        if (_p > p)
          p = _p;
      } }
      // binarization
      Bit b = (p < nc) ? -1 : 0;
      set_bit(bin_ref, n*So*So + r*So+c, b);
    } }
  }

  test_conv_layer(
      wt, kh, data_i, data_o, bin_ref,
      M, N, Si,   // n_outputs=4 to avoid assert failure in Accel
      1,  // conv_mode
      1   // max_pool
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

  test_conv_pool_layer_random( 8, wt, kh);
  test_conv_pool_layer_random(16, wt, kh);
  test_conv_pool_layer_random(32, wt, kh);

  delete[] wt;
  delete[] kh;

  printf ("Tests passed!\n");
  return 0;
}
