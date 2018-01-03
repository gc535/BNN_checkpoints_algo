//------------------------------------------------------------------------
// MaxLayer
//------------------------------------------------------------------------
#ifndef MAX_LAYER_H
#define MAX_LAYER_H

#include <cstddef>
#include <hls_video.h>
#include "BaseLayer.h"
#include "NormLayer.h"
#include "Typedefs.h"
  
//------------------------------------------------------------------------
// Max2NormLayer
//  - output map contains same number of maps, but each dim is shrinked
//  by factor of 2
//  - fused with the succeeding BatchNorm layer
//------------------------------------------------------------------------
template<unsigned N, unsigned S>
class Max2NormLayer : public SquareLayer<N,N,S,S>
{
  typedef SquareLayer<N,N,S,S> Super;
  static const unsigned S2 = S/2;

public:

  //SArray<KType, N> k;
  //SArray<HType, N> h;
  //KType* k;
  //HType* h;
  KType k[N];
  HType h[N];
  //--------------------------------------------------
  // Constructor
  //--------------------------------------------------
  Max2NormLayer(std::string s)
    : Super(s)
  {
    static_assert(S%2 == 0, "Max2Layer: S must be a multiple of 2\n");
    //k = new KType[N];
    //h = new HType[N];
  }
  ~Max2NormLayer() {
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
  template<typename InputArray, typename OutputArray>
  void get_output(const InputArray &in, OutputArray &out) const {
    
    typedef typename InputArray::ElemType  InputType;
    typedef typename OutputArray::ElemType OutputType;
    assert(in.size()  >= N*S*S);
    assert(out.size() >= N*S*S);

    for (unsigned n = 0; n < N; ++n) {


      MAX_ROWS: for (unsigned r = 0; r < S2; ++r) {
        MAX_COLS: for (unsigned c = 0; c < S2; ++c) {
          #pragma HLS PIPELINE
          InputType i00 = in[n*S*S + (2*r+0)*S + (2*c+0)];
          InputType i01 = in[n*S*S + (2*r+0)*S + (2*c+1)];
          InputType i10 = in[n*S*S + (2*r+1)*S + (2*c+0)];
          InputType i11 = in[n*S*S + (2*r+1)*S + (2*c+1)];

          InputType r0 = (i00 > i01) ? i00 : i01;
          InputType r1 = (i10 > i11) ? i10 : i11;
          InputType res = (r0 > r1) ? r0 : r1;

          NormOutput res_k = res * k[n];
          NormOutput res_h = res_k + h[n];

          out[n*S2*S2 + r*S2 + c] = sgn<NormOutput>( res_h );

        } // MAX_COLS
      } // MAX_ROWS

    }

  }

};

#endif
