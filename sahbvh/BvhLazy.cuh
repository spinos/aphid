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
	
	
}

#endif        //  #ifndef BVHLAZY_CUH

