#ifndef BVHLAZY_CUH
#define BVHLAZY_CUH

#include <cuda_runtime_api.h>
#include "bvh_math.cuh"
#include "radixsort_implement.h"
#include "Aabb.cuh"

__global__ void updateNodeAabbAtLevel_kernel(Aabb * nodeAabbs,
        int * nodeLevels,
        int2 * nodes,
        KeyValuePair * primitiveIndirections,
        Aabb * primitiveAabbs,
        int level,
	    uint n)
{
    uint ind = blockIdx.x*blockDim.x + threadIdx.x;
	if(ind >= n) return;
	
	if(nodeLevels[ind] != level) return;
	
	int2 child = nodes[ind];
	
	int i;
	Aabb box;
	resetAabb(box);
	if((child.x>>31)==0) {
        i=child.x;
	    for(;i<=child.y;i++)
	        expandAabb(box, primitiveAabbs[primitiveIndirections[i].value]);
	}
	else {
        child.x = getIndexWithInternalNodeMarkerRemoved(child.x);
        child.y = getIndexWithInternalNodeMarkerRemoved(child.y);
	    expandAabb(box, nodeAabbs[child.x]);
	    expandAabb(box, nodeAabbs[child.y]);
	}
	nodeAabbs[ind] = box;
}

#endif        //  #ifndef BVHLAZY_CUH

