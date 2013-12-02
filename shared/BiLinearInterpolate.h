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
	float interpolate(float u, float v, const float * src) const;
	Vector2F interpolate2(float u, float v, const Vector2F * src) const;
	void interpolate3(float u, float v, const Vector3F * src, Vector3F * dst) const;
};