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

// #define BVHSOLVER_DBG_DRAW 1

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

static bool isPow2(unsigned int x)
{
    return ((x&(x-1))==0);
}

struct Aabb {
    float3 low;
    float3 high;
	void combine(Aabb & a) {
		if(a.low.x < low.x) low.x = a.low.x;
		if(a.low.y < low.y) low.y = a.low.y;
		if(a.low.z < low.z) low.z = a.low.z;
		if(a.high.x > high.x) high.x = a.high.x;
		if(a.high.y > high.y) high.y = a.high.y;
		if(a.high.z > high.z) high.z = a.high.z;
	}
};

struct EdgeContact {
    uint v[4];   
};

#define MAX_INDEX 999999999

#endif        //  #ifndef BVH_COMMON_H

