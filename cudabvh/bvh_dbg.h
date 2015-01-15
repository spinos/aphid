#ifndef BVH_DBG_H
#define BVH_DBG_H

#include <iostream>
#include <sstream>
#include "bvh_common.h"

//cudaError_t err = cudaGetLastError();								\
//	if(err != cudaSuccess) std::cout<<"cuda error";
		

static const char * byte_to_binary(unsigned x)
{
    static char b[33];
    b[32] = '\0';

    for (int z = 0; z < 32; z++) {
        b[31-z] = ((x>>z) & 0x1) ? '1' : '0';
    }

    return b;
}

static const char * aabb_str(Aabb & a)
{   
    std::stringstream sst;
    sst<<"(("<<a.low.x<<","<<a.low.y<<","<<a.low.z<<"),("<<a.high.x<<","<<a.high.y<<","<<a.high.z<<"))";
    return sst.str().c_str();
}
#endif        //  #ifndef BVH_DBG_H

