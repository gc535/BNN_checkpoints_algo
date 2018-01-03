#include "InputConv.h"
#include "AccelTest.h"
#include "Timer.h"

const unsigned S = 32;
float input_conv_buffer[S*S];
float in_buffer[(S+2)*(S+2)];
static Timer t_iconv("input-conv");

void input_conv_layer_cpu(
    const uint64_t* w,
    const float* k,
    const float* h,
    const float* data_i,
    uint64_t* data_o,
    const unsigned M,
    const unsigned N
) {
  // Note that the input is stored non-interleaved,
  // so the i'th pixel of the m'th input image
  // is at index m*S*S + i

  t_iconv.start();

  // ------------------------------------------
  // assign 0 to the boundaries
  // ------------------------------------------
  for (unsigned c = 0; c < S+2; ++c)
    in_buffer[c] = 0;
  for (unsigned r = 1; r < S+1; ++r) {
    in_buffer[r*(S+2) + 0] = 0;
    in_buffer[r*(S+2) + S+1] = 0;
  }
  for (unsigned c = 0; c < S+2; ++c)
    in_buffer[(S+1)*(S+2) + c] = 0;

  // ------------------------------------------
  // Main conv loop
  // ------------------------------------------
  for (unsigned n = 0; n < N; ++n) {
    // clear input_conv_buffer
    for (unsigned i = 0; i < S*S; ++i) {
      input_conv_buffer[i] = 0;
    }

    // Loop over all input images
    for (unsigned m = 0; m < M; ++m) {
      unsigned w_idx = (n*M+m) / CONV_W_PER_WORD;
      unsigned w_off = (n*M+m) % CONV_W_PER_WORD;
      uint64_t w_word = w[w_idx] >> w_off*K*K;

      // copy input
      for (unsigned r = 1; r < S+1; ++r) {
        for (unsigned c = 1; c < S+1; ++c) {
          in_buffer[r*(S+2) + c] = data_i[m*S*S + (r-1)*S + (c-1)];
      } }

      // operate on 1 input image
      for (int r = 0; r < S; ++r) {
      for (int c = 0; c < S; ++c) {
        float res = 0;

        // perform convolution
        for (int kr = 0; kr < K; ++kr) {
        for (int kc = 0; kc < K; ++kc) {
          float pix = in_buffer[(r+kr)*(S+2) + (c+kc)];
          const int b = (w_word >> ((2-kr)*K+(2-kc))) & 0x1;
          res += (b!=0) ? -pix : pix;
        } } // kr,kc

        input_conv_buffer[r*S + c] += res;
      } } // r,c of input img
    } // m

    // perform batch-norm
    unsigned o_idx = n*S*S/WORD_SIZE;
    for (unsigned i = 0; i < S*S; i+=WORD_SIZE) {
      uint64_t o_word = 0;
      for (unsigned b = 0; b < WORD_SIZE; ++b) {
        uint64_t bit = (input_conv_buffer[i+b]*k[n] < -h[n]) ? 1 : 0;
        o_word = o_word | (bit << b);
      }
      data_o[o_idx++] = o_word;
    }
  } // n

  t_iconv.stop();
}
