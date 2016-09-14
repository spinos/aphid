#ifndef ALLMATH_H
#define ALLMATH_H
/*
 *  AllMath.h
 *  
 *
 *  Created by jian zhang on 4/11/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */
#include <cmath>
#include <stdlib.h>
#include <iostream>
#include <vector>
#include <deque>
#include <ATypes.h>
#include <Vector3F.h>
#include <Matrix33F.h>
#include <Matrix44F.h>
#include <Vector2F.h>

namespace aphid {

inline bool IsNan(float a) { return a != a; }

inline bool IsInf(float a) { return (a > 1e38 || a < -1e38); }

inline void Clamp01(float &v) {
	if(v < 0.f) v = 0.f;
	if(v > 1.f) v = 1.f;
}

#ifdef WIN32
inline float log2f( float n )  
{  
    return logf( n ) / logf( 2.f );  
}
#endif

#define PI 3.1415926535
#define EPSILON 10e-7
#define GoldenRatio 1.618
#define ReGoldenRatio 0.382

inline bool CloseToZero(float a) {
    return (a < 1e-4 && a > -1e-4);
}

inline int GetSign(float d) {
    if(d> 0.f) return 1;
    if(d< 0.f) return -1;
    return 0;
}

inline float RandomF01()
{ return ((float)(rand() & 1023)) * 0.0009765625f; }

inline float RandomFn11()
{ return (RandomF01() - 0.5f) * 2.f; }

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

template<typename T>
inline T Absolute(T const& a)
{
	return (a >= 0.0) ? a : -a;
}

template<typename T>
inline bool IsElementIn(T const& a, const std::vector<T>& array)
{
	typename std::vector<T>::const_iterator it;
	for(it = array.begin(); it != array.end(); ++it) {
		if(a == *it) return true;
	}
	return false;
}

template<typename T>
inline T DegreeToAngle(T const & a)
{
	return a * 3.14159269 / 180.0;
}

template<typename T>
inline T AngleToDegree(T const & a)
{
	return a / 3.14159269 * 180.0;
}

template<typename T>
inline char IsNearZero(T const & a)
{
	if(a > EPSILON || a < -EPSILON) return 0;
	return 1;
}

template<typename T>
inline void SwapAB(T & a, T & b, T & c)
{ c = a; a = b; b = c; }

template<typename T>
inline void ClampInPlace(T & a, const T & lowLimit, const T & highLimit)
{
	if(a < lowLimit) a = lowLimit;
	if(a > highLimit) a = highLimit;
}

template<typename T>
inline T MixClamp01F(const T & a, const T & b, const float & w)
{ 
	if(w < 0.f) 
		return a;
		
	if(w > 1.f)
		return b;
		
	return a * (1.f - w) + b * w;
}

template<typename T>
inline T RemapF(const T & a, const T & b, 
				const float & low, const float & high,
				const float & v)
{ 
	float w = (v - low) / (high - low);
	return a * (1.f - w) + b * w;
}

template<typename T>
inline void SameSign(T & a, const T & b)
{
	if(a * b < 0)
		a = -a;
}

}
#endif        //  #ifndef ALLMATH_H

