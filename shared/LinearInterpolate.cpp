/*
 *  LinearInterpolate.cpp
 *  knitfabric
 *
 *  Created by jian zhang on 6/5/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */

#include "LinearInterpolate.h"
namespace aphid {

LinearInterpolate::LinearInterpolate() {}

float LinearInterpolate::interpolate(float u, float * src) const
{
    return src[0] * (1.f - u) + src[1] * u;
}

}
