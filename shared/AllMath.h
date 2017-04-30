#ifndef APH_ALL_MATH_H
#define APH_ALL_MATH_H
/*
 *  AllMath.h
 *  
 *
 *  Created by jian zhang on 4/11/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */
#include <cstdlib>
#include <vector>
#include <deque>
#include <math/miscfuncs.h>
#include <math/Matrix44F.h>
#include <math/Ray.h>
#include <math/BarycentricCoordinate.h>
#include <math/BoundingBox.h>
#include <math/BoundingRectangle.h>
#include <math/AOrientedBox.h>
#include <math/Plane.h>
#include <math/PseudoNoise.h>
#include <math/ANoise3.h>
#include <math/MersenneTwister.h>

namespace aphid {

inline float closestDistanceToLine(const Vector3F * p, const Vector3F & toP, Vector3F & closestP, float & coord)
{
    Vector3F vr = toP - p[0];
    Vector3F v1 = p[1] - p[0];
	const float dr = vr.length();
	if(dr < 1e-5) {
        closestP = p[0];
        coord = 0.f;
		return 0.f;
    }
	
	const float d1 = v1.length();
	vr.normalize();
	v1.normalize();
	float vrdv1 = vr.dot(v1) * dr;
	if(vrdv1 < 0.f) vrdv1 = 0.f;
	if(vrdv1 > d1) vrdv1 = d1;
	
	v1 = p[0] + v1 * vrdv1;
	const float dc = v1.distanceTo(toP);
	
	closestP = v1;
	coord = vrdv1 / d1;
	return dc;
}

}
#endif        //  #ifndef ALLMATH_H

