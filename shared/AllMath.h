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
#include <boost/scoped_array.hpp>
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

inline float RandomF01()
{ return ((float)(rand() % 499))/499.f; }

inline float RandomFn11()
{ return (RandomF01() - 0.5f) * 2.f; }

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

#endif        //  #ifndef ALLMATH_H

