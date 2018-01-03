//------------------------------------------------------------------------
// ConvLayer
//------------------------------------------------------------------------
#ifndef CONV_LAYER_H
#define CONV_LAYER_H

#include <cstddef>
#include <hls_video.h>
#include "BaseLayer.h"
#include "Typedefs.h"

/*

//------------------------------------------------------------------------
// ConvHelper
// Contains all the functions needed to do a convolution efficiently
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
//------------------------------------------------------------------------

//#define INPUT_CONV_INT

// used to copy data into the input buffer
/*
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
* 
* *********************************************************************

template<unsigned M, unsigned N,
         unsigned S, unsigned K,
         unsigned _DB_LVL=5>
class ConvLayer : public SquareLayer<M,N,S,S>
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
  ConvLayer(std::string s)
    : Super(s)
  {
    //static_assert(K==3, "Conv layer only works for 3x3 convolutions!\n");
    //static_assert(M==3, "Input conv layer assumes 3 input channels!\n");
    w = new Bit[M*N*K*K];
    k = new float[N];
    h = new float[N];
  }
  ~ConvLayer() {
    delete[] h;
    delete[] k;
    delete[] w;
  }

  template<typename T>
  void load_weights(const T* weights) {
    for (unsigned i = 0; i < M*N*K*K; ++i) {
        w[i] = (weights[i] >= 0) ? 0 : 1; // Changed from ConvLayer2
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
    InputType pix = 0;
    // -------------------------
    // -1, 0, 1 ---> -1, 1
    // -------------------------
    for (int m = 0; m < M; m++){
		for (int r = 0; r < S; r++){
			for (int c = 0; c < S; c++){  
				pix = in[m*S*S + r*S + c] < 0 ? -1 : 1;
				in[m*S*S + r*S + c] = pix;
				//printf("input[%d] = %d, " (m*S*S + r*S + c), in[m*S*S + r*S + c]);
	}	}	}
     
    
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

          //conv_buffer[r*S + c] += res;
          out[n*S*S + r*S +c] += res;
        } } // r,c of input img
      } // m
	  /*
      // perform batch-norm
      for (unsigned i = 0; i < S*S; ++i) {
        #ifdef INPUT_CONV_INT
        float x = static_cast<float>(conv_buffer[i]) / FTOI_FACTOR;
        #else
        float x = static_cast<float>(conv_buffer[i]);
        #endif
        x = x * k[n] + h[n];
        out[n*S*S + i] = (x >= 0) ? 0 : -1;
      } *************
    } // n

  }

};

#endif
*/



 template<unsigned S, unsigned K,
          unsigned _DB_LVL=5>
 class ConvHelper
 {
 public:

   template <typename T>
   static T conv_mul(T in, Bit neg) {
     return (neg!=0) ? (T)(-in) : in;
   }

   //--------------------------------------------------
   // Initialize a linebuffer for a conv by filling with some zeros
   //--------------------------------------------------
   template <typename LineBuf>
   static void init_linebuffer(LineBuf &lbuf)
   {
    // #pragma HLS INLINE
     // Prologue: zero out the first K/2 linebuffers
     PROLOG_COLS: for (unsigned c = 0; c < S; ++c) {
       #pragma HLS UNROLL
       PROLOG_ROWS: for (unsigned r = 0; r < K/2; ++r) {
         #pragma HLS UNROLL
         lbuf.shift_up(c);
         lbuf.insert_top(0,c);
       }
     }
   }

   //--------------------------------------------------
   // Update window and line buffers with a new pixel
   // The type of 'pix' needs to match the types of
   // the window and linebuffer
   //--------------------------------------------------
   template<typename WindowElem,
            typename WindowBuf, typename LineBuf>
   static void add_pixel_to_buffers(
       const WindowElem pix,
       WindowBuf& win, LineBuf& lbuf,
       unsigned n, unsigned m,
       unsigned r, unsigned c)
   {
    // #pragma HLS INLINE
     // window: shift right, leaving rightmost column for new data
     win.shift_left();

     // window: fill top K-1 pixels of rightmost column from lbuf
     for (unsigned wr = 0; wr < K-1; ++wr) {
       #pragma HLS UNROLL
       WindowElem val = (c != S) ? lbuf(wr,c) : WindowElem(0);
       win.insert(val, wr,K-1);
     }

     // window: fill bottom right with new input pixel
     win.insert(pix, K-1,K-1);

     // lbuf: shift up column c
     if (c != S) {
       lbuf.shift_up(c);
       lbuf.insert_top(pix, c);
     }
   }

   //--------------------------------------------------
   // Optimized KxK convolution using a window buffer
   // w_n is which K*K weight kernel to use
   //--------------------------------------------------
   template<typename OutputType, typename WindowBuf, typename WtBuf>
   static void conv(OutputType &res, WindowBuf &win,
                      const  WtBuf &w, const unsigned w_n)
   { OutputType temp[]={0,0,0,0,0,0,0,0,0};
	//OutputType res2;	 
		  // std::cout<<"OUTTYPE= "<<sizeof(OutputType)<<std::endl;
     //#pragma HLS INLINE
     // perform convolution

     #pragma HLS UNROLL
     for (unsigned wr = 0; wr < K; ++wr) {
    // #pragma HLS UNROLL
       for (unsigned wc = 0; wc < K; ++wc) {
         //res+= conv_mul( win(wr,wc), w[w_n*K*K + (8-(wr*K+wc))] );
         temp[3*wr+wc] = conv_mul( win(wr,wc), w[w_n*K*K + (8-(wr*K+wc))] );
       }
     }
	 res=temp[0]+temp[1] +temp[2]+temp[3]+temp[4]+temp[5]+temp[6]+temp[7]+temp[8]; 

   }

 }; // class ConvHelper

 //------------------------------------------------------------------------
 // ConvLayer
 //  - output map size is always equal to input size
 //  - inputs and weights are binary {-1, +1}
 //  - temporary storage (linebuffers, etc) are {-1, 0, +1}
 //------------------------------------------------------------------------
 template<unsigned M, unsigned N,
          unsigned S, unsigned K,
          unsigned _DB_LVL=5>
 class ConvLayer : public SquareLayer<M,N,S,S>
 {
   typedef SquareLayer<M,N,S,S> Super;

 public:

   //SArray<Bit, N*M*K*K> w;
   Bit w[N*M*K*K];
    // w = new Bit[N*M*K*K];

    //w = Bit[N*M*K*K];
   //--------------------------------------------------
   // Constructor
   //--------------------------------------------------
   ConvLayer(std::string s)
     : Super(s)
   {
     static_assert(K==3, "Conv layer only works for 3x3 convolutions!\n");
    // w = new Bit[N*M*K*K];
   }
   ~ConvLayer() {
    // delete[] w;
   }

   template<typename T>
   void load_weights(const T* weights) {
     for (unsigned i = 0; i < M*N*K*K; ++i) {
       w[i] = (weights[i] >= 0) ? 0 : 1;
     }
   }

   //--------------------------------------------------
   // Binary conv layer output
   //--------------------------------------------------
   template<typename InputArray, typename OutputArray>
   void get_output(const InputArray &in, OutputArray &out) const {

     typedef typename InputArray::ElemType  InputType;
     typedef typename OutputArray::ElemType OutputType;
     typedef ConvHelper<S,K, _DB_LVL>   Conv;
     assert(in.size()  >= M*S*S);
     assert(out.size() >= N*S*S);

     hls::Window    <K,   K, TwoBit> win;
     hls::LineBuffer<K-1, S, TwoBit> lbuf;

     out.clear();


     for (unsigned n = 0; n < N; ++n) {
       for (unsigned m = 0; m < M; ++m) {

         unsigned w_n = n*M + m;

         Conv::init_linebuffer(lbuf);

         CONV_ROWS: for (unsigned r = 0; r < S+1; ++r) {
           CONV_COLS: for (unsigned c = 0; c < S+1; ++c) {

             #pragma HLS PIPELINE
             // The value of pix is determined from the pixel at [r,c]
             // 0 -> +1, -1 -> -1
             // or -> 0 for padding around the boundaries
             TwoBit pix = 0;
             if (r != S && c != S) {
               pix = in[m*S*S + r*S + c] < 0 ? -1 : 1;
             }

             Conv::add_pixel_to_buffers(pix, win, lbuf, n,m,r,c);

             if (r != 0 && c != 0) {
               OutputType res = 0;
               Conv::conv(res, win, w, w_n);
               out[n*S*S + (r-1)*S + (c-1)] += res;
             }
           } // CONV_COLS
         } // CONV_ROWS

         DB(_DB_LVL,
           if (n == 0 && m == 0) {
             printf ("\nm = %u\n", m);
             printf ("-- w[%u,%u] --\n", n,m);
             //w.print(w_n, K, 'i');
             //printf ("-- in[%u] --\n", m);
             in.print_sub(m, S, 8, 'i');
             printf ("-- out[%u] --\n", n);
             out.print_sub(n, S, 8, 'i');
             printf ("--\n");
           }
         );

       } // m
     } // n

   }

 };

 #endif
