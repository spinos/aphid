/*
 *  hilbertCurve.h
 *  
 *
 *  Created by jian zhang on 6/5/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 *  
 *      blue
 *      |
 *  1-------2
 *  |       |
 *  |   x0  | - red
 *  |       |
 *  0       3
 */
#ifndef APHID_HILBERT_CURVE_H
#define APHID_HILBERT_CURVE_H

namespace aphid {

inline int hilbert2DCoord( float x , float y , float x0 , float y0 , 
					float xRed , float yRed , float xBlue , float yBlue , 
					int dim )
{
	float c;
	int bits, res = 0;
	for ( int i = 0 ; i < dim ; ++i ) { 
		float coordRed = ( x - x0 ) * xRed + ( y - y0 ) * yRed ; 
		float coordBlue = ( x - x0 ) * xBlue + ( y - y0 ) * yBlue ; 
		xRed /= 2 ; yRed /= 2 ; xBlue /= 2 ; yBlue /= 2 ; 
		
		if ( coordRed <= 0 && coordBlue <= 0 ) { // quadr ant 0 
			x0 -= ( xBlue +xRed ) ; y0 -= ( yBlue +yRed ) ; 
			SwapAB<float> ( xRed , xBlue, c ) ; SwapAB<float> ( yRed , yBlue, c ) ; 
			bits = 0 ; 
		} 
		else if ( coordRed <= 0 && coordBlue >= 0 ) { // quadr ant 1 
			x0 += ( xBlue - xRed ) ; y0 += ( yBlue - yRed ) ; 
			bits = 1 ; 
		} 
		else if ( coordRed >= 0 && coordBlue >= 0 ) { // quadr ant 2 
			x0 += ( xBlue +xRed ) ; y0 += ( yBlue +yRed ) ; 
			bits = 2 ; 
		} 
		else if ( coordRed >= 0 && coordBlue <= 0 ) { // quadr ant 3 
			x0 += ( - xBlue +xRed ) ; y0 += ( - yBlue +yRed ) ; 
			SwapAB<float> ( xRed , xBlue, c ) ; SwapAB<float> ( yRed , yBlue, c ) ; 
			xBlue = -xBlue ; yBlue = -yBlue ; 
			xRed = -xRed ; yRed = -yRed ; 
			bits = 3; 
		} 
		res |= bits<<(28 - i*2);
	} 
	return res;
}

}
#endif
