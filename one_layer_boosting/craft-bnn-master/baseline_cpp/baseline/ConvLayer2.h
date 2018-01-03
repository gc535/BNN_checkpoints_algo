//------------------------------------------------------------------------
// ConvLayer
//------------------------------------------------------------------------
#ifndef CONV_LAYER2_H
#define CONV_LAYER2_H

#include <cstddef>
#include <hls_video.h>
#include "BaseLayer.h"
#include "Typedefs.h"
const int cp_lo0[128] = { 
//#include "checkpoints_mod/thresh_lo_filter_chk0.dat" 
#include "new_thresh/thresh_lo_filter_chk32sortalgo.txt" 
};
const int cp_lo1[128] = {
//#include "checkpoints_mod/thresh_lo_filter_chk1.dat" 
#include "new_thresh/thresh_lo_filter_chk64sortalgo.txt" 
};
const int cp_lo2[128] = { 
//#include "checkpoints_mod/thresh_lo_filter_chk2.dat" 
#include "new_thresh/thresh_lo_filter_chk96sortalgo.txt" 
};
const int cp_hi0[128] = { 
//#include "checkpoints_mod/thresh_hi_filter_chk0.dat" 
#include "new_thresh/thresh_hi_filter_chk32sortalgo.txt" 
};
const int cp_hi1[128] = { 
//#include "checkpoints_mod/thresh_hi_filter_chk1.dat" 
#include "new_thresh/thresh_hi_filter_chk64sortalgo.txt" 
};
const int cp_hi2[128] = { 
#include "new_thresh/thresh_hi_filter_chk96sortalgo.txt" 
};
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

 template<unsigned S, unsigned K,
          unsigned _DB_LVL=5>
 class Conv2Helper
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
 class Conv2Layer : public SquareLayer<M,N,S,S>
 {
   typedef SquareLayer<M,N,S,S> Super;

 public:
   int cp_lo[32*32*128];
   int cp_hi[32*32*128];
   //SArray<Bit, N*M*K*K> w;
   Bit w[N*M*K*K];
    // w = new Bit[N*M*K*K];

    //w = Bit[N*M*K*K];
   //--------------------------------------------------
   // Constructor
   //--------------------------------------------------
   Conv2Layer(std::string s)
     : Super(s)
   {
     static_assert(K==3, "Conv layer only works for 3x3 convolutions!\n");
    // w = new Bit[N*M*K*K];
   }
   ~Conv2Layer() {
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
     typedef Conv2Helper<S,K, _DB_LVL>   Conv;
     assert(in.size()  >= M*S*S);
     assert(out.size() >= N*S*S);

     hls::Window    <K,   K, TwoBit> win;
     hls::LineBuffer<K-1, S, TwoBit> lbuf;

     out.clear();

	 OutputType flgArr[S*S];
	 //flgArr.clear();


     for (unsigned n = 0; n < N; ++n) {
	   //printf("n = %d\n", n);
       for (unsigned m = 0; m < M; ++m) {

         unsigned w_n = n*M + m;

         Conv::init_linebuffer(lbuf);
		 //flgArr.clear();
		 //printf("clearing flag array\n");
		 for (int i = 0; i < S*S; i++){
		 	 flgArr[i] = 0;
		 }
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
			   if( N == 128){
                 if( m == 32 ){
				   //printf("enter checkpoint 0, m = %d\n", m);		 
			       if(out[n*S*S+(r-1)*S+(c-1)] < cp_lo0[n] || out[n*S*S+(r-1)*S+(c-1)] > cp_hi0[n]){
				     
				     //printf("change flag 0\n");		 
				     flgArr[S*(r-1) + (c-1)] = 1; 
					 //printf("exit checkpoint 0\n");
				   }   
				 }
				 if( m == 64){
				   //printf("enter checkpoint 1, m = %d\n", m);
			       if(out[n*S*S+(r-1)*S+(c-1)] < cp_lo1[n] || out[n*S*S+(r-1)*S+(c-1)] > cp_hi1[n]){
				     //printf("change flag 1\n");		 
				     flgArr[S*(r-1) + (c-1)] = 1;
					 //printf("exit checkpoint 1\n"); 
				   }   
				 }
				 if( m == 96){
			       //printf("enter checkpoint 2, m = %d\n", m);
			       if(out[n*S*S+(r-1)*S+(c-1)] < cp_lo2[n] || out[n*S*S+(r-1)*S+(c-1)] > cp_hi2[n]){ 
				    // printf("change flag 2\n");		 
				     flgArr[S*(r-1) + (c-1)] = 1;
					// printf("exit checkpoint 2\n"); 
				   }    
				 }

		       }	 				
			   		 
               if(flgArr[S*(r-1) + (c-1)] == 0){
			   OutputType res = 0;
               Conv::conv(res, win, w, w_n);
               out[n*S*S + (r-1)*S + (c-1)] += res;
			   }
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







/*template<unsigned S, unsigned K,
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
    #pragma HLS INLINE
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
    #pragma HLS INLINE
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
                      const WtBuf &w, const unsigned w_n)
  {
    #pragma HLS INLINE
    // perform convolution
    for (unsigned wr = 0; wr < K; ++wr) {
    #pragma HLS UNROLL
      for (unsigned wc = 0; wc < K; ++wc) {
        res += conv_mul( win(wr,wc), w[w_n*K*K + (8-(wr*K+wc))] );
      }
    }
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
class ConvLayer2 : public SquareLayer<M,N,S,S>
{
  typedef SquareLayer<M,N,S,S> Super;

public:

  //SArray<Bit, N*M*K*K> w;
  Bit *w;

  //--------------------------------------------------
  // Constructor
  //--------------------------------------------------
  ConvLayer2(std::string s)
    : Super(s)
  {
    static_assert(K==3, "Conv layer only works for 3x3 convolutions!\n");
    w = new Bit[N*M*K*K];
  }
  ~ConvLayer2() {
    delete[] w;
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

};*/

//#endif
