/*
 *  LinearInterpolate.h
 *  knitfabric
 *
 *  Created by jian zhang on 6/5/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */

#pragma once
#include <AllMath.h>
class LinearInterpolate {
public:
	LinearInterpolate();
	float interpolate(float u, float * src) const;
};
