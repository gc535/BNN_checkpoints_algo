//------------------------------------------------------------------------
// Normalization layers
//------------------------------------------------------------------------
#ifndef NORM_LAYER_H
#define NORM_LAYER_H

#include <ap_fixed.h>
#include "BaseLayer.h"
#include "Typedefs.h"
#include "Common.h"

template<unsigned N,  unsigned S>
class BatchNormLayer : public SquareLayer<N,N,S,S>
{
  typedef SquareLayer<N,N,S,S> Super;

public:
  
  //SArray<KType, N> k;
  //SArray<HType, N> h;
  KType k[N];
  HType h[N];

  //--------------------------------------------------
  // Constructor
  //--------------------------------------------------
  BatchNormLayer(std::string s)
    : Super(s)
  {
    //k = new KType[N];
    //h = new HType[N];
  }
  ~BatchNormLayer() {
    //delete[] k;
    //delete[] h;
  }
  
  template<typename T>
  void load_weights(const T* k_data, const T* h_data) {
    for (unsigned n = 0; n < N; ++n) {
      k[n] = k_data[n];
      h[n] = h_data[n];
    }
  }

  //--------------------------------------------------
  // Binary conv layer output

  template<typename InputArray, typename OutputArray>
  void get_output(const InputArray &in, OutputArray &out) const {

    typedef typename InputArray::ElemType InputType;
    assert(in.size()  >= N*S*S);
    assert(out.size() >= N*S*S);

    for (unsigned n = 0; n < N; ++n) {
      for (unsigned s = 0; s < S*S; ++s) {
      //  #pragma HLS PIPELINE

        NormOutput res_k = in[n*S*S + s] * k[n];
        NormOutput res_h = res_k + h[n];

        out[n*S*S + s] = sgn<NormOutput>( res_h );
      }
    }
  }
  
  template<typename InputArray, typename OutputArray>
  void get_output_float(const InputArray &in, OutputArray &out) const {

    typedef typename InputArray::ElemType InputType;
    assert(in.size()  >= N*S*S);
    assert(out.size() >= N*S*S);

    for (unsigned n = 0; n < N; ++n) {
      for (unsigned s = 0; s < S*S; ++s) {
        //#pragma HLS PIPELINE

        float res_k = in[n*S*S + s] * k[n].to_float();
        float res_h = res_k + h[n].to_float();

        out[n*S*S + s] = sgn<float>( res_h );
      }
    }
  }


};

#endif
