#ifndef BVHCOST_CUH
#define BVHCOST_CUH

#include <cuda_runtime_api.h>
#include "bvh_math.cuh"
#include "Aabb.cuh"

__global__ void computeTraverseCost_kernel(float * costs,
        int2 * nodes,
	    Aabb * nodeAabbs,
        uint n)
{
    uint ind = blockIdx.x*blockDim.x + threadIdx.x;
	if(ind >= n) return;
	
	costs[ind] = 0.f;
	
	int2 child = nodes[ind];
	if((child.x>>31) == 0) return;
	    
	Aabb leftBox = nodeAabbs[child.x];
	Aabb rightBox = nodeAabbs[child.y];
	Aabb rootBox = nodeAabbs[ind];
	
	costs[ind] = (areaOfAabb(&leftBox) + areaOfAabb(&rightBox))
	                / areaOfAabb(&rootBox);
}

#endif        //  #ifndef BVHCOST_CUH

