//------------------------------------------------------------------------
// Dense layers
//------------------------------------------------------------------------
#ifndef DENSE_LAYER_H
#define DENSE_LAYER_H

#include <ap_fixed.h>
#include "BaseLayer.h"
#include "Typedefs.h"
#define m1  0x55555555
#define m2  0x33333333
#define m4  0x0f0f0f0f
 
template<unsigned M, unsigned N>
class DenseLayer : public SquareLayer<M,N,1,1>
{
 // const unsigned m1 = 0x55555555;
 // const unsigned m2 = 0x33333333;
  //const unsigned m4 = 0x0f0f0f0f;
    
  typedef SquareLayer<M,N,1,1> Super;

public:
  
  const static unsigned UINT_BITS = sizeof(unsigned) * 8;
  const static unsigned M_TILE = 256;
  unsigned* w;
  KType* k;
  HType* h;

  //--------------------------------------------------
  // Constructor/Destructor
  //--------------------------------------------------
  DenseLayer(std::string s)
    : Super(s)
  {
    w = new unsigned[M*N/UINT_BITS];
    k = new KType[N];
    h = new HType[N];
  }

  ~DenseLayer() {
    delete[] w;
    delete[] k;
    delete[] h;
  }

  // The weights are originally laid out in M-major order,
  // which is really bad. We want to transpose the weights
  // as we load them to N-major orger
  template<typename T>
  void load_weights(const T* weights) {
    unsigned temp = 0;
    unsigned w_idx = 0;
    unsigned b = 0;

    for (unsigned n = 0; n < N; ++n) {
      for (unsigned m = 0; m < M; ++m) {
        if (weights[m*N + n] < 0) {
          temp |= (1 << b);
        }

        if (++b == UINT_BITS) {
          w[w_idx] = temp;
          temp = 0;
          ++w_idx;
          b = 0;
        }
      }
    }
  }

  // Load kh parameters
  template<typename T>
  void load_kh(const T* k_data, const T* h_data) {
    for (unsigned n = 0; n < N; ++n) {
      k[n] = k_data[n];
      h[n] = h_data[n];
    }
  }

  template<typename InputArray, typename OutputArray>
  void get_output(const InputArray &in, OutputArray &out) const {

    typedef typename InputArray::ElemType   InputType;
    typedef typename OutputArray::ElemType  OutputType;
    assert(in.size()  >= M);
    assert(out.size() >= N);
    assert(M % UINT_BITS == 0);
    assert(M_TILE % UINT_BITS == 0);

    /*for (unsigned n = 0; n < N; ++n) {
      out[n] = 0;
    }*/
    
    unsigned w_idx = 0;

    for (unsigned n = 0; n < N; ++n) {
      OutputType res = 0;

      for (unsigned m = 0; m < M; m+=UINT_BITS) {
        unsigned w_wrd = w[w_idx];
        unsigned d_wrd = 0;

        // pack 32 inputs into a word
        for (unsigned b = 0; b < UINT_BITS; ++b) {
          d_wrd |= ((in[m+b] == 0) ? 1 : 0) << b;
        }

        int x = w_wrd ^ d_wrd;

        // This gives 2*count_set_bit(x)
        x -= (x >> 1) & m1;
        x = (x & m2) + ((x >> 2) & m2);
        x = (x + (x >> 4)) & m4;
        x = (x * 0x01010101) >> 23;
        //x += x >> 8;
        //x += x >> 16;
        //x = (x & 0x7f) << 1;

        res += x - UINT_BITS;

        ++w_idx;
      }

      out[n] = res;
    }
  }
  
  /*template<typename InputArray, typename OutputArray>
  void get_output_kh(const InputArray &in, OutputArray &out) const {

    assert(in.size()  >= M);
    assert(out.size() >= N);
    assert(M % UINT_BITS == 0);

    unsigned w_idx = 0;

    for (unsigned n = 0; n < N; ++n) {
      ConvOutput res = 0;

      for (unsigned m = 0; m < M; m+=M_TILE) {
      for (unsigned mm = m; mm < m+M_TILE; mm+=UINT_BITS) {
        unsigned temp = w[w_idx];

        // do a batch of 32 using temp
        for (unsigned b = 0; b < UINT_BITS; ++b) {
          bool in_i = in[mm+b] == 0;     // True -> +1
          bool w_i  = (temp & (1 << b)) == 0;
          res += (in_i ^ w_i) ? -1 : 1;     // xnor
        }

        ++w_idx;
      }
      }

      NormOutput res_h = res * k[n] + h[n];
      out[n] = sgn<NormOutput>( res_h );
    }
  }*/

};

#endif
