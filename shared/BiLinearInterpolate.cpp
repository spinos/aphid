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

float BiLinearInterpolate::interpolate(float u, float v, float * src) const
{
    float lo = src[0] * (1.f - u) + src[1] * u;
    float hi = src[3] * (1.f - u) + src[2] * u;
    return lo * (1.f - v) + hi * v;
}