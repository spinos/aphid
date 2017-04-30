/*
 *  triangle_math.h
 *  
 *
 *  Created by jian zhang on 6/12/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef APHID_TRIANGLE_MATH_H
#define APHID_TRIANGLE_MATH_H

#include "AllMath.h"
namespace aphid {

inline bool insideTriangle(const Vector3F & p,
					const Vector3F & nor,
					const Vector3F & a,
					const Vector3F & b,
					const Vector3F & c)
{
	Vector3F e01 = b - a;
	Vector3F x0 = p - a;
	if(e01.cross(x0).dot(nor) < 0.f) return false;
	
	Vector3F e12 = c - b;
	Vector3F x1 = p - b;
	if(e12.cross(x1).dot(nor) < 0.f) return false;
	
	Vector3F e20 = a - c;
	Vector3F x2 = p - c;
	if(e20.cross(x2).dot(nor) < 0.f) return false;
	
	return true;
}

/// angle between ray and norm no larger than 90 deg
/// destination below origin
/// a b c clockwise 
inline bool segmentIntersectTriangle(const Vector3F & origin,
									const Vector3F & destination,
									const Vector3F & a,
									const Vector3F & b,
									const Vector3F & c)
{
	Vector3F ab = a - b;
	Vector3F cb = c - b;
	Vector3F nor = ab.cross(cb); nor.normalize();
	Vector3F de = destination - origin;
	float le = de.length();
	de /= le;
	float ddotn = de.dot(nor);
	float t = (a.dot(nor) - origin.dot(nor)) / ddotn;
	if(t<= 0.f) return false;
	aphid::Vector3F onplane = origin + de * t;
	
	return insideTriangle(onplane, nor, a, c, b);

}

}
#endif
