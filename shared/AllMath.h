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
#include <ATypes.h>
#include <Vector3F.h>
#include <Matrix33F.h>
#include <Matrix44F.h>
#include <Vector2F.h>

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

#endif        //  #ifndef ALLMATH_H

