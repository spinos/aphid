#include <cuda_runtime_api.h>
#include "bvh_math.cu"
#include "radixsort_implement.h"

namespace sahcompress {

__global__ void computeRunHead_kernel(uint * blockHeads, 
							KeyValuePair * mortonCode,
							uint d,
							uint maxElem,
							uint maxInd)
{
    unsigned ind = blockIdx.x*blockDim.x + threadIdx.x;
	if(ind >= maxInd) return;
	
	if(ind >= maxElem) {
	    blockHeads[ind] = 0;
	    return;
	}
	
	if(ind < 1) {
	    blockHeads[ind] = 1;
	    return;
	}

	uint clft = mortonCode[ind - 1].key;
	uint crgt = mortonCode[ind].key;
	
	if(clft>>d == crgt>>d) blockHeads[ind] = 0;
	else blockHeads[ind] = 1;
}

__global__ void compressRunHead_kernel(uint * compressed, 
							uint * runHeads,
							uint * indices,
							uint maxInd)
{
    unsigned ind = blockIdx.x*blockDim.x + threadIdx.x;
	if(ind >= maxInd) return;
	
	if(runHeads[ind]) compressed[indices[ind]] = ind;
}

__global__ void computeRunHash_kernel(KeyValuePair * compressed, 
						KeyValuePair * morton,
						uint * indices,
                        uint m,
						uint d,
						uint maxElem,
						uint maxInd)
{
    unsigned ind = blockIdx.x*blockDim.x + threadIdx.x;
	if(ind >= maxInd) return;
	
	compressed[ind].value = ind;
	
	if(ind>=maxElem)
	    compressed[ind].key = 1<<(m*3);
	else
	    compressed[ind].key = (morton[indices[ind]].key) >> d;
}

__global__ void computeRunLength_kernel(uint * runLength,
							uint * runHeads,
							uint nRuns,
							uint nPrimitives,
							uint maxInd)
{
    unsigned ind = blockIdx.x*blockDim.x + threadIdx.x;
	if(ind >= maxInd) return;
    
	if(ind >= nRuns) {
	    runLength[ind] = 0;
	    return;
	}

	if(ind >= nRuns-1) 
	    runLength[ind] = nPrimitives 
	                    - runHeads[ind];
	else
	    runLength[ind] = runHeads[ind+1] 
	                    - runHeads[ind];
}

__global__ void computeSortedRunLength_kernel(uint * runLength,
							uint * runHeads,
							KeyValuePair * indirections,
							uint nRuns,
							uint nPrimitives,
							uint maxInd)
{
    unsigned ind = blockIdx.x*blockDim.x + threadIdx.x;
	if(ind >= maxInd) return;
    
	if(ind >= nRuns) {
	    runLength[ind] = 0;
	    return;
	}
	
	uint sortedInd = indirections[ind].value;

	if(sortedInd >= nRuns-1) 
	    runLength[ind] = nPrimitives 
	                    - runHeads[sortedInd];
	else
	    runLength[ind] = runHeads[sortedInd+1] 
	                    - runHeads[sortedInd];
}
    
__global__ void computeClusterAabbs_kernel(Aabb * clusterAabbs,
            Aabb * primitiveAabbs,
            uint * runHeads,
            uint * runLength,
            uint numRuns)
{
    unsigned ind = blockIdx.x*blockDim.x + threadIdx.x;
	if(ind >= numRuns) return;
	
	const uint first = runHeads[ind];
	const uint l = runLength[ind];
	Aabb box;
    resetAabb(box);
    uint i = 0;
	for(;i<l;i++) 
        expandAabb(box, primitiveAabbs[first + i]);
	
    clusterAabbs[ind] = box;
}

__global__ void computeSortedClusterAabbs_kernel(Aabb * clusterAabbs,
            Aabb * primitiveAabbs,
            KeyValuePair * indirections,
            uint * runHeads,
            uint * runLength,
            uint numRuns)
{
    unsigned ind = blockIdx.x*blockDim.x + threadIdx.x;
	if(ind >= numRuns) return;
	
	const uint first = runHeads[ind];
	const uint l = runLength[ind];
	Aabb box;
    resetAabb(box);
    uint i = 0;
	for(;i<l;i++) 
        expandAabb(box, primitiveAabbs[indirections[first + i].value]);
	
    clusterAabbs[ind] = box;
}

}
