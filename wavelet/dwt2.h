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
//#include <linearMath.h>
#include "fltbanks.h"
#include <ATypes.h>

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

}

}
#endif
