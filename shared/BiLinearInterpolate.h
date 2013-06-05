/*
 *  BiLinearInterpolate.h
 *  knitfabric
 *
 *  Created by jian zhang on 6/5/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */

#pragma once
#include <AllMath.h>
class BiLinearInterpolate {
public:
	BiLinearInterpolate();
	float BiLinearInterpolate::interpolate(float u, float v, float * src) const;
};