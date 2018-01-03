#ifndef INPUT_CONV_LAYER_H
#define INPUT_CONV_LAYER_H

#include <cstddef>
#include <hls_video.h>
#include "BaseLayer.h"
#include "ConvLayer.h"
//#include "ConvLayer2.h"
#include "NormLayer.h"
#include "Typedefs.h"
  
//------------------------------------------------------------------------
// InputConvLayer
//  - output map size is always equal to input size
//  - inputs are float
//  - conv weights are binary (1 = -ve and 0 = +ve)
//  - we fuse conv and norm layers, the intermediate conv output is float
//------------------------------------------------------------------------
template<unsigned M, unsigned N,
         unsigned S, unsigned K,
         unsigned _DB_LVL=5>
class InputConvLayer : public SquareLayer<M,N,S,S>
{
  typedef SquareLayer<M,N,S,S> Super;

public:
  
  typedef float DataType;

  Bit* w;
  float* k;
  float* h;

  //--------------------------------------------------
  // Constructor
  //--------------------------------------------------
  InputConvLayer(std::string s)
    : Super(s)
  {
    static_assert(K==3, "Conv layer only works for 3x3 convolutions!\n");
    static_assert(M==3, "Input conv layer assumes 3 input channels!\n");
    w = new Bit[M*N*K*K];
    k = new float[N];
    h = new float[N];
  }
  ~InputConvLayer() {
    delete[] h;
    delete[] k;
    delete[] w;
  }

  template<typename T>
  void load_weights(const T* weights) {
    for (unsigned i = 0; i < M*N*K*K; ++i) {
      w[i] = (weights[i] < 0) ? -1 : 0;
    }
  }
  
  template<typename T>
  void load_kh(const T* k_data, const T* h_data) {
    for (unsigned n = 0; n < N; ++n) {
      k[n] = k_data[n];
      h[n] = h_data[n];
    }
  }

  //--------------------------------------------------
  // Main work function
  //--------------------------------------------------
  template<typename InputArray, typename OutputArray>
  void get_output(const InputArray &in, OutputArray &out) const {
    
    #pragma HLS ARRAY_PARTITION variable=w cyclic factor=M*K*K
    
    // For the input layer we fuse
    typedef typename InputArray::ElemType  InputType;
    typedef typename OutputArray::ElemType OutputType;
    typedef ConvHelper<S,K, _DB_LVL>   Conv;
    assert(in.size()  >= M*S*S);
    assert(out.size() >= N*S*S);

    /*hls::Window    <K,   K, InputType> win[M];
    hls::LineBuffer<K-1, S, InputType> lbuf[M];
    #pragma HLS ARRAY_PARTITION variable=win complete
    #pragma HLS ARRAY_PARTITION variable=lbuf complete*/
    hls::Window    <K,   K, InputType> win;
    hls::LineBuffer<K-1, S, InputType> lbuf;


    // stores the conv output pixels at [r,c] across M
    //DataType mbuf[M];
    //#pragma HLS ARRAY_PARTITION variable=mbuf complete
    
    DataType conv_buffer[S*S];

    for (unsigned n = 0; n < N; ++n) {
      // clear intermediate buffer
      for (unsigned i = 0; i < S*S; ++i)
        conv_buffer[i] = 0;

      for (unsigned m = 0; m < M; ++m) {
        
        unsigned w_n = n*M + m;
        
        PROLOG_COLS: for (unsigned c = 0; c < S; ++c) {
          #pragma HLS UNROLL
          PROLOG_ROWS: for (unsigned r = 0; r < K/2; ++r) {
            #pragma HLS UNROLL
            lbuf.shift_up(c);
            lbuf.insert_top(0, c);
          }
        }

        CONV_ROWS: for (unsigned r = 0; r < S+1; ++r) {
          CONV_COLS: for (unsigned c = 0; c < S+1; ++c) {
            #pragma HLS PIPELINE
            // The value of pix is determined from the pixel at [r,c]
            // 0 -> +1, -1 -> -1
            // or -> 0 for padding around the boundaries
            InputType pix = 0;
            if (r != S && c != S) {
              pix = in[m*S*S + r*S + c];
            }

            Conv::add_pixel_to_buffers(pix, win, lbuf, n,m,r,c);

            if (r != 0 && c != 0) {
              DataType res = 0;

              for (unsigned wr = 0; wr < K; ++wr) {
                for (unsigned wc = 0; wc < K; ++wc) {
                  const Bit& neg = w[w_n*K*K + (8-(wr*K+wc))];
                  const InputType val = win(wr,wc);
                  res += (neg!=0) ? (InputType)(-val) : val;
              } }

              conv_buffer[(r-1)*S + (c-1)] += res;
            }
          } // CONV_COLS
        } // CONV_ROWS

      } // m

      for (unsigned i = 0; i < S*S; ++i) {
        float x = static_cast<float>(conv_buffer[i]) * k[n] + h[n];
        out[n*S*S + i] = sgn<float>( x );
      }
    } // n

    // Parallelized across m, better for HLS
    /*for (unsigned n = 0; n < N; ++n) {

      // clear linebuffers for each new output map
      for (unsigned m = 0; m < M; ++m) {
        #pragma HLS UNROLL
        Conv::init_linebuffer(lbuf[m]);
      }

      CONV_ROWS: for (unsigned r = 0; r < S+1; ++r) {
        CONV_COLS: for (unsigned c = 0; c < S+1; ++c) {
          #pragma HLS PIPELINE
          
          // always update the conv buffers
          for (unsigned m = 0; m < M; ++m) {
            #pragma HLS UNROLL
            // The value of pix is determined from the pixel at [r,c]
            // 0 -> +1, -1 -> -1
            // or -> 0 for padding around the boundaries
            InputType pix = 0;
            if (r != S && c != S) {
              pix = in[m*S*S + r*S + c];
            }

            Conv::add_pixel_to_buffers(pix, win[m], lbuf[m], n,m,r,c);
          } // m

          // only perform the conv and store if legal position
          if (r != 0 && c != 0) {
            for (unsigned m = 0; m < M; ++m) {
              #pragma HLS UNROLL
              unsigned w_n = n*M + m;
              DataType res = 0;
              Conv::conv(res, win[m], w, w_n);
              mbuf[m] = res;
            }
              
            // sum the results across M
            DataType sum = 0;
            for (unsigned m = 0; m < M; ++m) {
              #pragma HLS UNROLL
              sum += mbuf[m];
            }

            // perform normalization right here
            out[n*S*S + (r-1)*S + (c-1)] =
              sgn<DataType>( sum * k[n] + h[n] );
          }

        } // CONV_COLS
      } // CONV_ROWS

    } // n
    */
  }

};

#endif
