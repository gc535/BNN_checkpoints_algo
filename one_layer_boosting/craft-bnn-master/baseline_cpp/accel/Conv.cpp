#include "Conv.h"
#include "Timer.h"
#include "AccelPrint.h"

int conv_buffer[DMEM_WORDS*WORD_SIZE];
static Timer t_conv("bin-conv");
static Timer t_pool("pool");

// binary convolution
void bin_conv(
    const uint64_t* w, const uint64_t* in,
    int* conv_out,
    unsigned M, unsigned N,
    unsigned S
) {
  for (int n = 0; n < (int)N; ++n) {
    for (int i = 0; i < S*S; ++i) {
      conv_out[n*S*S + i] = 0;
    }

    for (int m = 0; m < (int)M; ++m) {
      unsigned w_idx = (n*M+m) / CONV_W_PER_WORD;
      unsigned w_off = (n*M+m) % CONV_W_PER_WORD;
      uint64_t w_word = w[w_idx] >> w_off*K*K;

      // SxS image
      for (int r = 0; r < S; ++r) {
      for (int c = 0; c < S; ++c) {
        int res = 0;

        // 3x3 kernel
        for (int kr = 0; kr < K; ++kr) {
        for (int kc = 0; kc < K; ++kc) {
          int pix = 0;
          int _r = r+kr-K/2, _c = c+kc-K/2;
          if (_r >= 0 && _c >= 0 && _r < S && _c < S) {
            unsigned in_idx = (m*S*S + _r*S) / WORD_SIZE;
            unsigned in_off = (_r*S + _c) % WORD_SIZE;
            pix = ((in[in_idx] >> in_off) & 1LL) == 0 ? 1 : -1;
          }

          int b = (w_word >> ((2-kr)*K+(2-kc))) & 0x1;
          res += (b!=0) ? -pix : pix;
        } } // 3x3 kernel

        conv_out[n*S*S + r*S + c] += res;
      } } // SxS image
    } // M
  } // N
}

// batch norm binarization
void bin_conv_binarize(
    const float* k,
    const float* h,
    const int* conv_res,
    uint64_t* bin_out,
    unsigned N,
    unsigned S
) {
  unsigned o_idx = 0;
  for (unsigned n = 0; n < N; ++n) {
    for (unsigned i = 0; i < S*S; i+=WORD_SIZE) {
      uint64_t o_word = 0;
      for (unsigned b = 0; b < WORD_SIZE; ++b) {
        uint64_t bit = (conv_res[n*S*S+i+b] * k[n] < -h[n]) ? 1 : 0;
        o_word = o_word | (bit << b);
      }
      bin_out[o_idx++] = o_word;
  } }
}

void bin_conv_binarize_pool(
    const float* k,
    const float* h,
    const int* conv_res,
    uint64_t* bin_out,
    unsigned N,
    unsigned Si
) {
  uint64_t o_word = 0;
  unsigned o_idx = 0;
  unsigned o_off = 0;
  unsigned p_idx = 0; // idx to top-left of the 2x2 pooling window

  const unsigned So = Si/POOL_WIDTH;
  for (unsigned n = 0; n < N; ++n) {
    for (unsigned r = 0; r < So; ++r) {
      for (unsigned c = 0; c < So; ++c) {
        //unsigned idx = n*Si*Si + POOL_WIDTH*r*Si + POOL_WIDTH*c;
        int p = conv_res[p_idx];
        // max pooling
        for (unsigned pr = 0; pr < POOL_WIDTH; ++pr) {
        for (unsigned pc = 1; pc < POOL_WIDTH; ++pc) {
          int _p = conv_res[p_idx + pr*Si + pc];
          p = (_p > p) ? _p : p;
        } }
        p_idx += POOL_WIDTH;

        // binarization
        uint64_t b = (p*k[n] < -h[n]) ? 1 : 0;
        o_word = o_word | (b << o_off);
        o_off++;
      } // So cols

      if (o_off == WORD_SIZE) {
        bin_out[o_idx++] = o_word;
        o_off = 0;
        o_word = 0;
      }

      p_idx += Si;

    } // So rows
  }
}

void bin_conv_layer_cpu(
    const uint64_t* w,
    const float* k,
    const float* h,
    const uint64_t* data_i,
    uint64_t* data_o,
    const unsigned M,
    const unsigned N,
    const unsigned width,
    const unsigned max_pool
) {
  t_conv.start();

  bin_conv(
    w, data_i,
    conv_buffer,
    M, N, width
  );

  t_conv.stop();

  //printf ("*** SW conv result ***\n");
  //print_mat(conv_buffer, width, 8, width);

  t_pool.start();

  if (!max_pool) {
    bin_conv_binarize(
      k, h,
      conv_buffer,
      data_o,
      N, width
    );
  } else {
    bin_conv_binarize_pool(
      k, h,
      conv_buffer,
      data_o,
      N, width
    );
  }

  t_pool.stop();
}
