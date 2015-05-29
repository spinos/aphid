#ifndef BROADPHASE_CUH
#define BROADPHASE_CUH

#include "stripedModel.cu"
#include "bvh_math.cuh"
#include "radixsort_implement.h"
#include "Aabb.cuh"

#define B3_BROADPHASE_MAX_STACK_SIZE 64
#define B3_BROADPHASE_MAX_STACK_SIZE_M_2 62

inline __device__ int isStackFull(int stackSize)
{return stackSize > B3_BROADPHASE_MAX_STACK_SIZE_M_2; }

inline __device__ int outOfStack(int stackSize)
{return (stackSize < 1 || stackSize > B3_BROADPHASE_MAX_STACK_SIZE); }

inline __device__ void writeElementExclusion(int * dst,
									uint a,
									uint * exclusionInd,
									uint * exclusionStart)
{
	uint i;

	uint cur = exclusionStart[a+1]-1;
	uint minInd = exclusionStart[a];
// for isolate element
	if(minInd == cur) { 
	    dst[0] = exclusionInd[cur]; 
	    dst[1] = -1; 
	    return; 
	}
	
	for(i=0;i<32; i++) {
		dst[i] = exclusionInd[cur--];
		if(cur < minInd) {
		    if(i<31) {
		        dst[i+1] = -1;
		    }
		    return; 
		}
	}
}

inline __device__ int isElementExcludedS(uint b, int * exclusionInd)
{
	uint i;
	for(i=0; i<32; i++) {
		if(exclusionInd[i] < 0) break;
		if(b <= exclusionInd[i]) return 1;
	}
	return 0;
}

inline __device__ void countOverlappings(uint & count,
                                         KeyValuePair * indirections,
                                         Aabb box,
                                         Aabb * elementBoxes,
                                         int * exclElm,
                                         int2 range)
{
    uint iElement;
    int i=range.x;
    for(;i<=range.y;i++) {
        iElement = indirections[i].value;
        if(isElementExcludedS(iElement, exclElm)) continue;
        if(isAabbOverlapping(box, elementBoxes[iElement])) 
            count++;
    }
}

inline __device__ void writeOverlappings(uint2 * overlappings,
                                uint & writeLoc,
                                uint iQuery,
                                uint iBox,
                                uint startLoc,
                                uint cacheSize,
                                KeyValuePair * indirections,
                                Aabb box,
                                Aabb * elementBoxes,
                                int * exclElm,
                                int2 range)
{
    uint2 pair;
    uint iElement;
    int i=range.x;
    for(;i<=range.y;i++) {
        iElement = indirections[i].value;
        if(isElementExcludedS(iElement, exclElm)) continue;
        if(isAabbOverlapping(box, elementBoxes[iElement])) 
        {
            pair.x = combineObjectElementInd(iQuery, iBox);
			pair.y = combineObjectElementInd(iQuery, iElement);
			overlappings[writeLoc] = pair;
            writeLoc++;
        }
        if((writeLoc - startLoc)==cacheSize) return;
    }
}

inline __device__ int isInternalNode(int2 child)
{ return (child.x>>31) != 0; }

__global__ void countPairsSExclS_kernel(uint * overlappingCounts, 
								Aabb * boxes, 
								uint maxBoxInd,
								int2 * internalNodeChildIndices, 
								Aabb * internalNodeAabbs, 
								Aabb * leafAabbs,
								KeyValuePair * mortonCodesAndAabbIndices,
								uint * exclusionIndices,
								uint * exclusionStarts)
{
	// int * sStack = SharedMemory<int>();
	
	__shared__ int sExclElm[32*32];
	__shared__ int sStack[64*32];
	
	uint boxIndex = blockIdx.x*blockDim.x + threadIdx.x;
	if(boxIndex >= maxBoxInd) return;
	
	const Aabb box = boxes[boxIndex];
	
	int * exclElm = & sExclElm[threadIdx.x << 5];
	writeElementExclusion(exclElm, boxIndex, exclusionIndices,
									exclusionStarts);
	
	int * stack = &sStack[threadIdx.x << 6];
	int stackSize = 1;
	stack[0] = 0x80000000;
		
	int isInternal;
    int iNode;
    int2 child;
    Aabb internalBox;
	
    uint iCount = 0;
	for(;;) {
		if(outOfStack(stackSize)) break;
		
		iNode = stack[ stackSize - 1 ];
		stackSize--;
		
		// isLeaf = isLeafNode(internalOrLeafNodeIndex);	//Internal node if false
		iNode = getIndexWithInternalNodeMarkerRemoved(iNode);
        child = internalNodeChildIndices[iNode];
        isInternal = isInternalNode(child);
        
        if(!isInternal) {
            countOverlappings(iCount,
                                mortonCodesAndAabbIndices,
                                box,
                                leafAabbs,
                                exclElm,
                                child);
            continue;
        }
				
        internalBox = internalNodeAabbs[iNode];
        
//bvhRigidIndex is not used if internal node
		//int bvhRigidIndex = (isLeaf) ? mortonCodesAndAabbIndices[bvhNodeIndex].value : -1;
		
		//Aabb bvhNodeAabb = (isLeaf) ? leafAabbs[bvhRigidIndex] : internalNodeAabbs[bvhNodeIndex];

		if(isAabbOverlapping(box, internalBox))
		{    
/*		    
			if(isLeaf)
			{
			    // if(!isElementExcluded(bvhRigidIndex, boxIndex, exclusionIndices, exclusionStarts)) 
				if(!isElementExcludedS(bvhRigidIndex, exclElm))
			        overlappingCounts[boxIndex] += 1;
			}
			else*/ 
				if(isStackFull(stackSize)) continue;
			    
                stack[ stackSize ] = child.x;
                stackSize++;
                stack[ stackSize ] = child.y;
                stackSize++;
		}
	}
    overlappingCounts[boxIndex] = iCount;
}

__global__ void writePairCacheSExclS_kernel(uint2 * dst, 
                                uint * cacheWriteLocation,
								uint * cacheStarts, 
								uint * overlappingCounts,
								Aabb * boxes, 
								uint maxBoxInd,
								int2 * internalNodeChildIndices, 
								Aabb * internalNodeAabbs, 
								Aabb * leafAabbs,
								KeyValuePair * mortonCodesAndAabbIndices,
								unsigned queryIdx,
								uint * exclusionIndices,
								uint * exclusionStarts)
{
	__shared__ int sExclElm[32*32];
	__shared__ int sStack[64*32];
	
	uint boxIndex = blockIdx.x*blockDim.x + threadIdx.x;
	if(boxIndex >= maxBoxInd) return;
	
	uint cacheSize = overlappingCounts[boxIndex];
	if(cacheSize < 1) return;
	
	uint startLoc = cacheStarts[boxIndex];
	uint writeLoc = cacheWriteLocation[boxIndex];
	
	if((writeLoc - startLoc) >= cacheSize) return;
	
	const Aabb box = boxes[boxIndex];
	
	int * exclElm = & sExclElm[threadIdx.x << 5];
	writeElementExclusion(exclElm, boxIndex, exclusionIndices,
									exclusionStarts);
	
	int * stack = &sStack[threadIdx.x << 6];
	int stackSize = 1;
	stack[0] = 0x80000000;
		
	int isInternal;
    int iNode;
    int2 child;
    Aabb internalBox;
	
	for(;;) {
		if(outOfStack(stackSize)) break;
        
		iNode = stack[ stackSize - 1 ];
		stackSize--;
		
		iNode = getIndexWithInternalNodeMarkerRemoved(iNode);
		child = internalNodeChildIndices[iNode];
        isInternal = isInternalNode(child);
			    
        if(!isInternal) {
           writeOverlappings(dst,
                                writeLoc,
                                queryIdx,
                                boxIndex,
                                startLoc,
                                cacheSize,
                                mortonCodesAndAabbIndices,
                                box,
                                leafAabbs,
                                exclElm,
                                child);
           continue;
        }
        
		internalBox = internalNodeAabbs[iNode];
        
		if(isAabbOverlapping(box, internalBox))
		{
			/*if(isLeaf)
			{
			    // if(!isElementExcluded(bvhRigidIndex, boxIndex, exclusionIndices, exclusionStarts)) {
				if(!isElementExcludedS(bvhRigidIndex, exclElm)) {
			        pair.x = combineObjectElementInd(queryIdx, boxIndex);
			        pair.y = combineObjectElementInd(queryIdx, bvhRigidIndex);
			        dst[writeLoc] = pair;
			        writeLoc++;
			    }
			    
			    if((writeLoc - startLoc)==cacheSize) { // cache if full
			        break;
			    }
			}
			else */
				if(isStackFull(stackSize)) continue;
			    
                stack[ stackSize ] = child.x;
                stackSize++;
                stack[ stackSize ] = child.y;
                stackSize++;
		}
	}
	cacheWriteLocation[boxIndex] = writeLoc;
}

__global__ void startAsWriteLocation_kernel(uint * dst, uint * src, uint maxInd)
{
    unsigned ind = blockIdx.x*blockDim.x + threadIdx.x;

	if(ind >= maxInd) return;
	dst[ind] = src[ind];
}

#endif        //  #ifndef BROADPHASE_CUH

