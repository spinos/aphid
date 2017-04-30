/*
 *  HermiteCurve.cpp
 *  softIk
 *
 *  Created by jian zhang on 3/24/15.
 *  Copyright 2015 __MyCompanyName__. All rights reserved.
 *
 *  Reference:
 *  http://cubic.org/docs/hermite.htm
 */

#include "HermiteCurve.h"

HermiteCurve::HermiteCurve() {}
HermiteCurve::~HermiteCurve() {}

Vector3F HermiteCurve::interpolate(const float & s) const
{
	float s2 = s * s;
	float s3 = s2 * s;
	float h1 =  2.f * s3 - 3.f * s2 + 1.f;          // calculate basis function 1
	float h2 = -2.f * s3 + 3.f * s2;              // calculate basis function 2
	float h3 =   s3 - 2.f * s2 + s;         // calculate basis function 3
	float h4 =   s3 -  s2;              // calculate basis function 4
	Vector3F p = _P[0] * h1 +                    // multiply and sum all funtions
             _P[1] * h2 +                    // together to build the interpolated
             _T[0] * h3 +                    // point along the curve.
             _T[1] * h4;
	return p;
}
