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

static const float LegendreNormWeightF[11] = {
1.f,
1.f,
.6666667f,
.4f,
.22857143f,
.12698413f,
.06926407f,
.037296037f,
.01989122f,
.01053064f,
.005542445f
};

/// http://web.cs.iastate.edu/~cs577/handouts/orthogonal-polys.pdf
/// evalute 0 - n th legendre polynomials through interval [a, b] 
/// m number of evaluation nodes
/// v length of m * (n+1)
void legendreRules(int m, int n, float * v,
		float a, float b);

/// linear interpolate between knots over interval [a, b]
/// nknots a lot smaller than m
float interpolate(int nknots, const float * yknots, 
	int m, float x, float a, float b, float dx);

/// http://mathfaculty.fullerton.edu/mathews/n2003/TrapezoidalRuleMod.html
/// discrete integrate over interal [a, b], m equal spacing nodes
/// finding the area under a curve y = f(x) over [a, b] with m subintervals x0, x1, xm-1
/// x0 = a, xm-1 = b
float trapezIntegral(const float & a, const float & b, 
	int m, const float * y);

/// http://mathfaculty.fullerton.edu/mathews/n2003/SimpsonsRule2DMod.html
/// http://mathfaculty.fullerton.edu/mathews/n2003/GaussianQuadMod.html
/// https://en.wikipedia.org/wiki/Adaptive_Simpson%27s_method
/// http://www.math.pitt.edu/~sussmanm/2070Fall07/lab_10/index.html
/// http://ab-initio.mit.edu/wiki/index.php/Cubature
/// http://math2.uncc.edu/~shaodeng/TEACHING/math5172/Lectures/Lect_15.PDF
/// https://people.sc.fsu.edu/~jburkardt/cpp_src/product_rule/product_rule.html

}

}