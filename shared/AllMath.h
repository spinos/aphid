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
#include <Vector3F.h>
#include <Matrix33F.h>
#include <Matrix44F.h>
#include <Vector2F.h>
#include <Ray.h>

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

#endif        //  #ifndef ALLMATH_H

