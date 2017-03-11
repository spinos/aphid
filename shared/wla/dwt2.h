/*
 *  dwt2.h
 *  
 *
 *  Created by jian zhang on 9/15/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 */
#ifndef WLA_DWT_2D_H
#define WLA_DWT_2D_H

#include "fltbanks.h"

namespace aphid {

namespace wla {

/// X[N] input signal
/// P phase of shift |P| < N
/// delay the signal when P > 0
void circleShift1(float * x, const int & n, const int & p);

/// circshift row index for all columns of x by p
/// example 
/// x has 5 rows p = 2
/// y = x(3:4:0:1:2, :) 
void circleShift2(Array2<float> * x, const int & p);

/// analysis filter bank per column
/// resulting subband of half input number of rows
/// x m-by-n input signal
/// lo output low pass subband
/// hi output high pass subband
void afbRow(const Array2<float> * x, Array2<float> * lo, Array2<float> * hi);
void afbRowflt(const Array2<float> * x,
		const float flt[2][10],
		Array2<float> * lo, Array2<float> * hi);

/// synthesis filter bank per column
/// input lowpass and highpass subbands
/// output y doubles the number of rows of input
void sfbRow(Array2<float> * y, Array2<float> * lo, Array2<float> * hi);
void sfbRowflt(Array2<float> * y, 
		const float flt[2][10],
		Array2<float> * lo, Array2<float> * hi);

/// http://eeweb.poly.edu/iselesni/WaveletSoftware/standard2D.html
/// 2d analysis filter bank
/// filter along columns of x
/// transpose the resulting 2 subbands
/// filter again
/// resulting 4 subbands half of the size of input signal x
void afb2flt(const Array2<float> * x,
		const float flt[2][10],
		Array2<float> * lo, Array2<float> * lohi,
		Array2<float> * hilo, Array2<float> * hihi);

/// 2d synthesis filter banks
/// 4 input subbands
/// output y doubles the size of input
void sfb2flt(Array2<float> * y,
		const float flt[2][10],
		Array2<float> * lo, Array2<float> * lohi,
		Array2<float> * hilo, Array2<float> * hihi);
		
}

}
#endif
