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
#include <iostream>
#include <cuda_runtime_api.h>

typedef unsigned int uint;
typedef unsigned long long uint64;

typedef float4 BarycentricCoordinate;

struct ClosestPointTestContext {
    float3 closestPoint;
    float closestDistance;
};

struct ContactData {
    float4 separateAxis;
    float3 localA;
    float padding;
    float3 localB;
    float timeOfImpact;
};

struct mat33 {
    float3 v[3];
};

struct mat44 {
    float4 v[4];
};

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

static uint iRound1024(uint n)
{
	return iDivUp(n, 1024) * 1024;
}

static int iLog2(int n)
{
	int i=0;
	while(n > (1<<i)) {
		i++;
	}
	return i;
}

struct EdgeContact {
    uint v[4];   
};

struct RayInfo {
    float3 origin;
    float3 destiny;
};

#define MAX_INDEX 2147483647
#define TINY_VALUE 1e-10
#define TINY_VALUE2 1e-8
#define HUGE_VALUE 1e12

struct __align__(4) Aabb {
    float3 low;
    float3 high;
};

struct __align__(16) Aabb4 {
    float4 low;
    float4 high;
};

#endif        //  #ifndef BVH_COMMON_H

