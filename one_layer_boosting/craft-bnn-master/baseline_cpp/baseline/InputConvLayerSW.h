#ifndef INPUT_CONV_LAYER_SW_H
#define INPUT_CONV_LAYER_SW_H

#include <cstddef>
#include "Typedefs.h"
#include "BaseLayer.h"

//#define INPUT_CONV_INT

// used to copy data into the input buffer
template<typename T1>
void copy_input(T1& input, const float* data) {
  #ifdef INPUT_CONV_INT
  // copy input and multiply by a large factor
  for (unsigned i = 0; i < input.size(); ++i) {
    input[i] = data[i] * ConvLayer1::FTOI_FACTOR;
  }
  #else
  input.copy_from(data);
  #endif
}

template<unsigned M, unsigned N,
         unsigned S, unsigned K,
         unsigned _DB_LVL=5>
class InputConvLayerSW : public SquareLayer<M,N,S,S>
{
  typedef SquareLayer<M,N,S,S> Super;

public:

#ifdef INPUT_CONV_INT
  typedef int DataType;
  static constexpr float FTOI_FACTOR = 1000000;
#else
  typedef float DataType;
#endif

  Bit* w;
  float* k;
  float* h;

  //--------------------------------------------------
  // Constructor
  //--------------------------------------------------
  InputConvLayerSW(std::string s)
    : Super(s)
  {
    static_assert(K==3, "Conv layer only works for 3x3 convolutions!\n");
    static_assert(M==3, "Input conv layer assumes 3 input channels!\n");
    w = new Bit[M*N*K*K];
    k = new float[N];
    h = new float[N];
  }
  ~InputConvLayerSW() {
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

    DataType conv_buffer[S*S];
    DataType in_buffer[(S+2)][(S+2)];

    // ------------------------------------------
    // assign 0 to the boundaries
    // ------------------------------------------
    for (unsigned c = 0; c < S+2; ++c)
      in_buffer[0][c] = 0;
    for (unsigned r = 1; r < S+1; ++r) {
      in_buffer[r][0] = 0;
      in_buffer[r][S+1] = 0;
    }
    for (unsigned c = 0; c < S+2; ++c)
      in_buffer[S+1][c] = 0;

    // ------------------------------------------
    // Main conv loop
    // ------------------------------------------
    for (unsigned n = 0; n < N; ++n) {
      // clear conv_buffer
      for (unsigned i = 0; i < S*S; ++i) {
        conv_buffer[i] = 0;
      }

      for (unsigned m = 0; m < M; ++m) {
        const unsigned w_n = n*M + m;

        // copy input
        for (unsigned r = 1; r < S+1; ++r) {
          for (unsigned c = 1; c < S+1; ++c) {
            in_buffer[r][c] = in[m*S*S + (r-1)*S + (c-1)];
        } }

        // work on 1 input image
        for (int r = 0; r < S; ++r) {
        for (int c = 0; c < S; ++c) {
          DataType res = 0;

          for (int kr = 0; kr < K; ++kr) {
          for (int kc = 0; kc < K; ++kc) {
            DataType pix = in_buffer[r+kr][c+kc];
            const Bit b = w[w_n*K*K + (8-(kr*K+kc))];

            res += (b==0) ? pix : -pix;
          } } // kr,kc

          conv_buffer[r*S + c] += res;
        } } // r,c of input img
      } // m

      // perform batch-norm
      for (unsigned i = 0; i < S*S; ++i) {
        #ifdef INPUT_CONV_INT
        float x = static_cast<float>(conv_buffer[i]) / FTOI_FACTOR;
        #else
        float x = static_cast<float>(conv_buffer[i]);
        #endif
        x = x * k[n] + h[n];
        out[n*S*S + i] = (x >= 0) ? 0 : -1;
      }
    } // n

  }

};

#endif
