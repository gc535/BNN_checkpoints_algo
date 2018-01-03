#include <cstddef>
#include <hls_video.h>

const unsigned S = 8;
const unsigned K = 3;

void top(
    const unsigned img[S*S],
    const unsigned kernel[K*K],
    unsigned out[S*S]
){
  #pragma HLS ARRAY_PARTITION variable=kernel

  // 3x3 window for convolution
  hls::Window<K,K, unsigned> win;
  hls::LineBuffer<K-1,S, unsigned> lbuf;

  // Prologue: zero out the first K/2 linebuffers
  PROLOG_COLS: for (unsigned c = 0; c < S; ++c) {
    #pragma HLS UNROLL
    PROLOG_ROWS: for (unsigned r = 0; r < K/2; ++r) {
      #pragma HLS UNROLL
      lbuf.shift_up(c);
      lbuf.insert_top(0,c);
    }
  }
  
  // Main loop:
  //  (r,c) is where to load from input.
  //  (_r,_c) is where to store to output, equal to (r-1,c-1)
  //  Loop r,c from 0 .. S instead of 0 .. S-1
  //
  // Vertical boundary handling:
  //  When c==S, do not load, shift a zero column into win
  //  When c==0, do not perform computation
  //
  // Horizontal boundary handling:
  //  When r==0, do not perform computation
  //  When r==S, do not load, shift a zero into win
  CONV_ROWS: for (unsigned r = 0; r < S+1; ++r) {
    CONV_COLS: for (unsigned c = 0; c < S+1; ++c) {
      #pragma HLS PIPELINE
      // read a new input pixel at [r,c]
      unsigned pix = 0;
      if (r != S && c != S)
        pix = img[r*S+c];
      
      // window: shift right, leaving rightmost column for new data
      win.shift_left();
      
      // window: fill top 2 pixels of rightmost column from lbuf
      for (unsigned wr = 0; wr < K-1; ++wr) {
        #pragma HLS UNROLL
        unsigned val = (c != S) ? lbuf(wr,c) : 0;
        win.insert(val, wr,K-1);
      }
      
      // window: fill bottom right with new input pixel
      win.insert(pix, K-1,K-1);

      // lbuf: shift up column c
      if (c != S) {
        lbuf.shift_up(c);
        lbuf.insert_top(pix, c);
      }

      // perform convolution
      if (r != 0 && c != 0) {
        unsigned res = 0;
        for (unsigned wr = 0; wr < K; ++wr) {
          for (unsigned wc = 0; wc < K; ++wc) {
            res += win(wr,wc) * kernel[wr*K+wc];
          }
        }
        out[(r-1)*S+(c-1)] = res;
      }
    }
  }
 
}

int main() {
  unsigned img[S*S];
  unsigned kernel[K*K];
  unsigned out[S*S];

  for (unsigned r = 0; r < S; ++r)
    for (unsigned c = 0; c < S; ++c) {
      img[r*S + c] = r+c;
      out[r*S + c] = 0;
    }

  for (unsigned kk = 0; kk < K*K; ++kk)
    kernel[kk] = 0;
  kernel[8] = 2;

  // run
  top(img, kernel, out);

  // print
  printf ("img:\n");
  for (unsigned r = 0; r < S; ++r) {
    for (unsigned c = 0; c < S; ++c)
      printf ("%3u ", img[r*S+c]);
    printf ("\n");
  }
  printf ("kernel:\n");
  for (unsigned r = 0; r < K; ++r) {
    for (unsigned c = 0; c < K; ++c)
      printf ("%3u ", kernel[r*K+c]);
    printf ("\n");
  }
  printf ("out:\n");
  for (unsigned r = 0; r < S; ++r) {
    for (unsigned c = 0; c < S; ++c)
      printf ("%3u ", out[r*S+c]);
    printf ("\n");
  }

  return 0;
}
