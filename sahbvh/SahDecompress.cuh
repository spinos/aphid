#include <cuda_runtime_api.h>
#include "bvh_common.h"
#include "radixsort_implement.h"

namespace sahdecompress {
    
__global__ void initHash_kernel(KeyValuePair * primitiveIndirections,
                    uint n)
{
    uint ind = blockIdx.x*blockDim.x + threadIdx.x;
	if(ind >= n) return;
	
	primitiveIndirections[ind].value = ind;
}

__global__ void countLeaves_kernel(uint * leafLengths,
                    int * qelements,
                    int2 * nodes,
                    KeyValuePair * indirections,
                    uint * runHeads,
                    uint numHeads,
                    uint numPrimitives,
                    uint numNodes,
                    uint scanLength)
{
    uint ind = blockIdx.x*blockDim.x + threadIdx.x;
	if(ind >= scanLength) return;
	
	if(ind >= numNodes) {
	    leafLengths[ind] = 0;
	    return;
	}
	
	int2 range = nodes[ind];
	if((range.x >> 31)) {
	    leafLengths[ind] = 0;
	    return;
	}
	
	uint sum = 0;
	uint group;
	uint runLength;
	int i = range.x;
	for(;i<= range.y; i++) {
	    group = indirections[i].value;
	    if(group < numHeads-1)
	        runLength = runHeads[group+1] - runHeads[group];
	    else 
	        runLength = numPrimitives - runHeads[group];
	    sum +=runLength;
	}
	leafLengths[ind] = sum;
}

__global__ void copyHash_kernel(KeyValuePair * dst,
					KeyValuePair * src,
					uint maxInd)
{
    unsigned ind = blockIdx.x*blockDim.x + threadIdx.x;
	if(ind >= maxInd) return;
	
	dst[ind] = src[ind];
}

__global__ void decompressIndices_kernel(uint * decompressedIndices,
                    uint * compressedIndices,
					KeyValuePair * sorted,
					uint * offset,
					uint * runLength,
					uint nRuns)
{
    unsigned ind = blockIdx.x*blockDim.x + threadIdx.x;
	if(ind >= nRuns) return;
	
	const uint sortedInd = sorted[ind].value;
	const uint start = offset[ind];
	const uint first = compressedIndices[sortedInd];
	const uint l = runLength[ind];
	
	uint i = 0;
	for(;i<l;i++)
	    decompressedIndices[start + i] = first + i;
}

__global__ void decompressPrimitives_kernel(KeyValuePair * dst,
                            KeyValuePair * src,
                            int2 * nodes,
                            KeyValuePair * indirections,
                            uint* leafOffset,
                            uint * runHeads,
                            uint numHeads,
                            uint numPrimitives,
                            uint maxInd)
{
    unsigned ind = blockIdx.x*blockDim.x + threadIdx.x;
	if(ind >= maxInd) return;
	
	int2 range = nodes[ind];
	if((range.x >> 31))
	    return;
	
	uint group;
	uint groupBegin;
	uint runLength;
	uint writeInd = leafOffset[ind];
	int i, j;
	for(i = range.x;i<= range.y; i++) {
	    group = indirections[i].value;
	    if(group < numHeads-1)
	        runLength = runHeads[group+1] - runHeads[group];
	    else 
	        runLength = numPrimitives - runHeads[group];
	    
	    groupBegin = runHeads[group];
	    for(j=0; j<runLength; j++) {
	        dst[writeInd] = src[groupBegin+j];
	        writeInd++;
	    }
	}
	
	range.x = leafOffset[ind];
	range.y = writeInd-1;
	nodes[ind] = range;
}

__global__ void writeSortedHash_kernel(KeyValuePair * dst,
							KeyValuePair * src,
							uint * indices,
							uint maxInd)
{
    unsigned ind = blockIdx.x*blockDim.x + threadIdx.x;
	if(ind >= maxInd) return;
	
	dst[ind] = src[indices[ind]];
}

}
