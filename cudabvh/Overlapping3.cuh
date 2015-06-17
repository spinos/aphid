#ifndef OVERLAPPING3_CUH
#define OVERLAPPING3_CUH

#include "stackUtil.cuh"
#include "bvhUtil.h"

#define EXCLU_IN_SMEM 0

template<int NumExcls>
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
        if(isElementExcludedS<NumExcls>(iElement, exclElm)) continue;
        if(isAabbOverlapping(box, elementBoxes[iElement])) 
            count++;
    }
}

template<int NumExcls>
inline __device__ void writeOverlappings(uint2 * overlappings,
                                uint & writeLoc,
                                uint iQuery,
                                uint iBox,
                                KeyValuePair * indirections,
                                Aabb box,
                                Aabb * elementBoxes,
                                int * exclElm,
                                int2 range)
{
    uint2 pair;
    pair.x = combineObjectElementInd(iQuery, iBox);
    uint iElement;
    int i=range.x;
    for(;i<=range.y;i++) {
        iElement = indirections[i].value;
        if(!isElementExcludedS<NumExcls>(iElement, exclElm)) {
            if(isAabbOverlapping(box, elementBoxes[iElement])) {
                pair.y = combineObjectElementInd(iQuery, iElement);
                overlappings[writeLoc] = pair;
                writeLoc++;
            }
        }
    }
}

template<int NumExcls, int NumThreads>
__global__ void countPairsSelfCollideSingle_kernel(uint * overlappingCounts, 
								Aabb * boxes, 
								uint maxBoxInd,
								int2 * internalNodeChildIndices, 
								Aabb * internalNodeAabbs, 
								Aabb * leafAabbs,
								KeyValuePair * mortonCodesAndAabbIndices,
								int * exclusionIndices)
{
    int *sdata = SharedMemory<int>();
		
	uint boxIndex = blockIdx.x*blockDim.x + threadIdx.x;
	if(boxIndex >= maxBoxInd) return;
	
	const Aabb box = boxes[boxIndex];

#if 1
    int * exclElm = &sdata[threadIdx.x * NumExcls];
    writeElementExclusion<NumExcls>(exclElm, &exclusionIndices[boxIndex * NumExcls]);
#else
	int * exclElm = &exclusionIndices[boxIndex*NumExcls];
#endif	
	
	int stack[BVH_TRAVERSE_MAX_STACK_SIZE];
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
		
		iNode = getIndexWithInternalNodeMarkerRemoved(iNode);
        child = internalNodeChildIndices[iNode];
        isInternal = isInternalNode(child);
		
        internalBox = internalNodeAabbs[iNode];

		if(isAabbOverlapping(box, internalBox))
		{    
		    if(!isInternal) {
		        countOverlappings<NumExcls>(iCount,
                                mortonCodesAndAabbIndices,
                                box,
                                leafAabbs,
                                exclElm,
                                child);
            }
            else {
				if(isStackFull(stackSize)) continue;
			    
                stack[ stackSize ] = child.x;
                stackSize++;
                stack[ stackSize ] = child.y;
                stackSize++;
            }
		}
	}
    overlappingCounts[boxIndex] = iCount;
}

template<int NumExcls, int NumThreads>
__global__ void writePairCacheSelfCollideSingle_kernel(uint2 * dst, 
                                uint * cacheWriteLocation,
								uint * overlappingCounts,
								Aabb * boxes, 
								uint maxBoxInd,
								int2 * internalNodeChildIndices, 
								Aabb * internalNodeAabbs, 
								Aabb * leafAabbs,
								KeyValuePair * mortonCodesAndAabbIndices,
								unsigned queryIdx,
								int * exclusionIndices)
{
    int *sdata = SharedMemory<int>();
	
	uint boxIndex = blockIdx.x*blockDim.x + threadIdx.x;
	if(boxIndex >= maxBoxInd) return;
	
	uint cacheSize = overlappingCounts[boxIndex];
	if(cacheSize < 1) return;
	
	uint writeLoc = cacheWriteLocation[boxIndex];
	
	const Aabb box = boxes[boxIndex];
	
#if 1
    int * exclElm = &sdata[threadIdx.x * NumExcls];
    writeElementExclusion<NumExcls>(exclElm, &exclusionIndices[boxIndex * NumExcls]);
#else
	int * exclElm = &exclusionIndices[boxIndex*NumExcls];
#endif
	
	int stack[BVH_TRAVERSE_MAX_STACK_SIZE];
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
        
		internalBox = internalNodeAabbs[iNode];
        
		if(isAabbOverlapping(box, internalBox))
		{
			if(!isInternal) {
			    writeOverlappings<NumExcls>(dst,
                                writeLoc,
                                queryIdx,
                                boxIndex,
                                mortonCodesAndAabbIndices,
                                box,
                                leafAabbs,
                                exclElm,
                                child);
            }
			else {
				if(isStackFull(stackSize)) continue;
			    
                stack[ stackSize ] = child.x;
                stackSize++;
                stack[ stackSize ] = child.y;
                stackSize++;
            }
		}
	}
	cacheWriteLocation[boxIndex] = writeLoc;
}
#endif        //  #ifndef OVERLAPPING3_CUH

