#ifndef BVHCOST_CUH
#define BVHCOST_CUH

#include <cuda_runtime_api.h>
#include "bvh_math.cuh"
#include "Aabb.cuh"

__global__ void computeTraverseCost_kernel(float * costs,
        int2 * nodes,
        int * nodeNumPrimitives,
	    Aabb * nodeAabbs,
        uint n)
{
    uint ind = blockIdx.x*blockDim.x + threadIdx.x;
	if(ind >= n) return;
	
	costs[ind] = 0.f;
	
	int2 child = nodes[ind];
	if((child.x>>31) == 0) return;
    
    child.x = getIndexWithInternalNodeMarkerRemoved(child.x);
    child.y = getIndexWithInternalNodeMarkerRemoved(child.y);
	    
	Aabb leftBox = nodeAabbs[child.x];
	Aabb rightBox = nodeAabbs[child.y];
	Aabb rootBox = nodeAabbs[ind];
	
	costs[ind] = 2.5f + (areaOfAabb(&leftBox) * nodeNumPrimitives[child.x] 
	                + areaOfAabb(&rightBox) * nodeNumPrimitives[child.y])
	                / areaOfAabb(&rootBox);
}

__global__ void countPrimitviesInNodeAtLevel_kernel(int * nodeNumPrimitives,
        int * nodeLevels,
        int2 * nodes,
        int level,
	    uint n)
{
    uint ind = blockIdx.x*blockDim.x + threadIdx.x;
	if(ind >= n) return;
	
	if(nodeLevels[ind] != level) return;
    
	int2 child = nodes[ind];
	
	if((child.x>>31)==0) {
        nodeNumPrimitives[ind] = child.y - child.x + 1;
	}
	else {
        child.x = getIndexWithInternalNodeMarkerRemoved(child.x);
        child.y = getIndexWithInternalNodeMarkerRemoved(child.y);
	    nodeNumPrimitives[ind] = nodeNumPrimitives[child.x] + nodeNumPrimitives[child.y];
	}
}

#endif        //  #ifndef BVHCOST_CUH

