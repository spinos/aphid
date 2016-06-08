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
 *
 *  0   -1 -1
 *  1   -1  1
 *  2    1  1
 *  3    1 -1
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

/*
 *     z blue
 *     |   y green
 *     |  /
 *     | /
 *     |/______x red
 *
 *      2      5
 *     / |   / |
 *    1  |   6 |
 *    |  3---|-4
 *    |      |
 *    0      7
 *
 *    0  -1 -1 -1
 *    1  -1 -1  1
 *    2  -1  1  1
 *    3  -1  1 -1
 *    4   1  1 -1
 *    5   1  1  1
 *    6   1 -1  1
 *    7   1 -1 -1
 */
inline int hilbert3DCoord( float x , float y, float z,
					float x0 , float y0 , float z0,
					float xRed , float yRed , float zRed,
					float xGreen, float yGreen, float zGreen,
					float xBlue , float yBlue , float zBlue,
					int dim )
{
	float c;
	int bits, res = 0;
	for ( int i = 0 ; i < dim ; ++i ) { 
		float coordRed = ( x - x0 ) * xRed + ( y - y0 ) * yRed + ( z - z0 ) * zRed ; 
		float coordGreen = ( x - x0 ) * xGreen + ( y - y0 ) * yGreen + ( z - z0 ) * zGreen; 
		float coordBlue = ( x - x0 ) * xBlue + ( y - y0 ) * yBlue + ( z - z0 ) * zBlue; 
		xRed /= 2 ; yRed /= 2; zRed /=2;
		xGreen /= 2; yGreen /= 2; zGreen /=2;
		xBlue /= 2 ; yBlue /= 2 ; zBlue /= 2;
		
		if(coordRed <= 0) {
			if(coordGreen <= 0) {
				if(coordBlue <= 0) { 
/// octant 0 
					x0 -= ( xGreen+xBlue +xRed ) ; 
					y0 -= ( yGreen+yBlue +yRed ) ; 
					z0 -= zGreen + zBlue + zRed;
					SwapAB<float> ( xRed , xGreen, c ) ; 
					SwapAB<float> ( yRed , yGreen, c ) ; 
					SwapAB<float> ( zRed , zGreen, c ) ; 
					SwapAB<float> ( xRed , xBlue, c ) ; 
					SwapAB<float> ( yRed , yBlue, c ) ; 
					SwapAB<float> ( zRed , zBlue, c ) ;
					bits = 0 ; 
				}
				else {
/// octant 1
					x0 += -xRed - xGreen + xBlue; 
					y0 += -yRed - yGreen + yBlue;
					z0 += -zRed - zGreen + zBlue; 
					SwapAB<float> ( xRed , xBlue, c ) ; 
					SwapAB<float> ( yRed , yBlue, c ) ; 
					SwapAB<float> ( zRed , zBlue, c ) ; 
					SwapAB<float> ( xRed , xGreen, c ) ; 
					SwapAB<float> ( yRed , yGreen, c ) ; 
					SwapAB<float> ( zRed , zGreen, c ) ; 
					bits = 1; 
				}
			}
			else {
				if(coordBlue > 0) {
/// octant 2		
					x0 += -xRed + xGreen + xBlue; 
					y0 += -yRed + yGreen + yBlue;
					z0 += -zRed + zGreen + zBlue; 
					SwapAB<float> ( xRed , xBlue, c ) ; 
					SwapAB<float> ( yRed , yBlue, c ) ; 
					SwapAB<float> ( zRed , zBlue, c ) ; 
					SwapAB<float> ( xRed , xGreen, c ) ; 
					SwapAB<float> ( yRed , yGreen, c ) ; 
					SwapAB<float> ( zRed , zGreen, c ) ; 
					bits = 2;
				}
				else {
/// octant 3 
					x0 += -xRed + xGreen - xBlue; 
					y0 += -yRed + yGreen - yBlue;
					z0 += -zRed + zGreen - zBlue; 
					xGreen = -xGreen; yGreen = -yGreen; zGreen = -zGreen;
					xBlue = -xBlue; yBlue = -yBlue; zBlue = -zBlue;
					bits = 3;
				}
			}
		}
		else {
			if(coordGreen > 0) {
				if(coordBlue <= 0) { 
/// octant 4		
					x0 += xRed + xGreen - xBlue; 
					y0 += yRed + yGreen - yBlue;
					z0 += zRed + zGreen - zBlue; 
					xGreen = -xGreen; yGreen = -yGreen; zGreen = -zGreen;
					xBlue = -xBlue; yBlue = -yBlue; zBlue = -zBlue;
					bits = 4;
				}
				else { 
/// octant 5		
					x0 += xRed + xGreen + xBlue; 
					y0 += yRed + yGreen + yBlue;
					z0 += zRed + zGreen + zBlue;
					
					SwapAB<float> ( xRed , xBlue, c ) ; 
					SwapAB<float> ( yRed , yBlue, c ) ; 
					SwapAB<float> ( zRed , zBlue, c ) ; 
					SwapAB<float> ( xRed , xGreen, c ) ; 
					SwapAB<float> ( yRed , yGreen, c ) ; 
					SwapAB<float> ( zRed , zGreen, c ) ;
					xRed = -xRed; yRed = -yRed; zRed = -zRed;
					xBlue = -xBlue; yBlue = -yBlue; zBlue = -zBlue;
					
					bits = 5;
				}
			}
			else {
				if (coordBlue > 0) { 
/// octant 6			
					x0 += xRed - xGreen + xBlue; 
					y0 += yRed - yGreen + yBlue;
					z0 += zRed - zGreen + zBlue;
					
					SwapAB<float> ( xRed , xBlue, c ) ; 
					SwapAB<float> ( yRed , yBlue, c ) ; 
					SwapAB<float> ( zRed , zBlue, c ) ; 
					SwapAB<float> ( xRed , xGreen, c ) ; 
					SwapAB<float> ( yRed , yGreen, c ) ; 
					SwapAB<float> ( zRed , zGreen, c ) ;
					xRed = -xRed; yRed = -yRed; zRed = -zRed;
					xBlue = -xBlue; yBlue = -yBlue; zBlue = -zBlue;
					
					bits = 6;
				}
				else {
/// octant 7	
					x0 += xRed - xGreen - xBlue; 
					y0 += yRed - yGreen - yBlue;
					z0 += zRed - zGreen - zBlue;
					
					SwapAB<float> ( xRed , xBlue, c ) ; 
					SwapAB<float> ( yRed , yBlue, c ) ; 
					SwapAB<float> ( zRed , zBlue, c ) ; 
					SwapAB<float> ( xGreen , xBlue, c ) ; 
					SwapAB<float> ( yGreen , yBlue, c ) ; 
					SwapAB<float> ( zGreen , zBlue, c ) ;
					xRed = -xRed; yRed = -yRed; zRed = -zRed;
					xGreen = -xGreen; yGreen = -yGreen; zGreen = -zGreen;
					
					bits = 7;
				}
			}
		}
		
		res |= bits<<(27 - i*3);
	} 
	return res;
}

}
#endif
