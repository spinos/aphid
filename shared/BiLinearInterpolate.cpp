/*
 *  BiLinearInterpolate.cpp
 *  knitfabric
 *
 *  Created by jian zhang on 6/5/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */

#include "BiLinearInterpolate.h"

BiLinearInterpolate::BiLinearInterpolate() {}

float BiLinearInterpolate::interpolate(float u, float v, const float * src) const
{
    float lo = src[0] * (1.f - u) + src[1] * u;
    float hi = src[3] * (1.f - u) + src[2] * u;
    return lo * (1.f - v) + hi * v;
}

Vector2F BiLinearInterpolate::interpolate2(float u, float v, const Vector2F * src) const
{
	Vector2F lo = src[0] * (1.f - u) + src[1] * u;
    Vector2F hi = src[3] * (1.f - u) + src[2] * u;
    return lo * (1.f - v) + hi * v;
}

void BiLinearInterpolate::interpolate3(float u, float v, const Vector3F * src, Vector3F * dst) const
{
	Vector3F lo = src[0] * (1.f - u) + src[1] * u;
    Vector3F hi = src[3] * (1.f - u) + src[2] * u;
    *dst = lo * (1.f - v) + hi * v;
}