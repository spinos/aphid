/*
 *  Calculus.h
 *  foo
 *
 *  Created by jian zhang on 8/21/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 */
#pragma once
#include "AllMath.h"

namespace aphid {

namespace calc {

/// evalute 0 - n th legendre polynomials through interval [a, b] 
/// m number of evaluation nodes
/// v length of m * (n+1)
inline void legendreRules(int m, int n, float * v,
							float a, float b)
{
	const float dx = (b - a) / (m - 1);
  int i;
  int j;

  for ( i = 0; i < m; i++ )
  {
    v[i+0*m] = 1.0;
  }

  for ( i = 0; i < m; i++ )
  {
    v[i+1*m] = a + dx * i;
  }
 
  for ( j = 2; j <= n; j++ )
  {
    for ( i = 0; i < m; i++ )
    {
      v[i+j*m] = ( ( float ) ( 2 * j - 1 ) *  (a + dx * i) * v[i+(j-1)*m]   
                 - ( float ) (     j - 1 ) *        v[i+(j-2)*m] ) 
                 / ( float ) (     j     );
    }
  }
}

/// linear interpolate between knots over interval [a, b]
/// nknots a lot smaller than m
inline float interpolate(int nknots, const float * yknots, int m, float x, float a, float b, float dx)
{
	int g = (x - a) / dx;
	if(g<1)
		return yknots[0];
	if(g>=m-1)
		return yknots[nknots-1];
		
	float movern =  ((float)m - 1.0) / ((float)nknots - 1);
	int n0 = g / movern;
	int n1 = n0 + 1;
	
	float alpha = (g - n0 * movern) / movern;
	return yknots[n0] * (1.f - alpha) + yknots[n1] * alpha;
}

/// http://mathfaculty.fullerton.edu/mathews/n2003/TrapezoidalRuleMod.html
/// discrete integrate over interal [a, b], m equal spacing nodes
/// finding the area under a curve y = f(x) over [a, b] with m subintervals x0, x1, xm-1
/// x0 = a, xm-1 = b
inline float trapezIntegral(const float & a, const float & b, int m, const float * y)
{
	const float h = (b - a) / (m - 1);
	float c = h * .5f * (y[0] + y[m-1]);
	int i=1;
	for(;i<m-1;++i) {
		c += h * y[i];
	}
	return c;
}

}

}