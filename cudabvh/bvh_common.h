#ifndef BVH_COMMON_H
#define BVH_COMMON_H

/*
 *  bvh_common.h
 *  
 *
 *  Created by jian zhang on 1/9/15.
 *  Copyright 2015 __MyCompanyName__. All rights reserved.
 *
 */
#include <cuda_runtime_api.h>
typedef unsigned int uint;

static uint iDivUp(uint dividend, uint divisor)
{
    return ( (dividend % divisor) == 0 ) ? (dividend / divisor) : (dividend / divisor + 1);
}

static uint nextPow2( uint x ) {
    --x;
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    return ++x;
}

struct Aabb {
    float3 low;
    float3 high;
};

#endif        //  #ifndef BVH_COMMON_H

