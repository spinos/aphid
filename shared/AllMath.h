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

#endif        //  #ifndef ALLMATH_H

