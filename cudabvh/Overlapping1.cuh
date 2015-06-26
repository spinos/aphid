#ifndef OVERLAPPING1_CUH
#define OVERLAPPING1_CUH

/* 
 *  collision single traverse
 */

#include "stackUtil.cuh"
#include "bvhUtil.cuh"

inline __device__ void countOverlappings(uint & count,
                                         KeyValuePair * indirections,
                                         Aabb box,
                                         Aabb * elementBoxes,
                                         int2 range)
{
    uint iElement;
    int i=range.x;
    for(;i<=range.y;i++) {
        iElement = indirections[i].value;
        if(isAabbOverlapping(box, elementBoxes[iElement])) 
            count++;
    }
}

inline __device__ void writeOverlappings(uint2 * overlappings,
                                uint & writeLoc,
                                uint iQuery,
                                uint iBox,
                                Aabb box,
                                uint iTree,
                                int2 range,
                                KeyValuePair * indirections,
                                Aabb * elementBoxes)
{
    uint2 pair;
    pair.x = combineObjectElementInd(iQuery, iBox);
    uint iElement;
    int i=range.x;
    for(;i<=range.y;i++) {
        iElement = indirections[i].value;
        if(isAabbOverlapping(box, elementBoxes[iElement])) {
            pair.y = combineObjectElementInd(iTree, iElement);
            overlappings[writeLoc] = pair;
            writeLoc++;
        }
    }
}

template<int NumSkip>
__global__ void countPairsSingle_kernel(uint * overlappingCounts, 
								Aabb * boxes, 
								uint maxBoxInd,
								int2 * internalNodeChildIndices, 
								Aabb * internalNodeAabbs, 
								Aabb * leafAabbs,
								KeyValuePair * mortonCodesAndAabbIndices)
{
    uint boxInd = blockIdx.x*blockDim.x + threadIdx.x;
    boxInd = boxInd<<NumSkip;
	if(boxInd >= maxBoxInd) return;
	
	const Aabb box = boxes[boxInd];	
	
	int stack[BVH_TRAVERSE_MAX_STACK_SIZE];
	int stackSize = 1;
	stack[0] = 0x80000000;
		
	int isLeaf;
    int iNode;
    int2 child;
    Aabb internalBox;
	
    uint outCount = overlappingCounts[boxInd];
	for(;;) {
		if(outOfStack(stackSize)) break;
		
		iNode = stack[ stackSize - 1 ];
		stackSize--;
		
		iNode = getIndexWithInternalNodeMarkerRemoved(iNode);
        child = internalNodeChildIndices[iNode];
        isLeaf = isLeafNode(child);
		
        internalBox = internalNodeAabbs[iNode];

		if(isAabbOverlapping(box, internalBox))
		{    
		    if(isLeaf) {
		        countOverlappings(outCount,
                                mortonCodesAndAabbIndices,
                                box,
                                leafAabbs,
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
    overlappingCounts[boxInd] = outCount;
}

template<int NumSkip>
__global__ void writePairCacheSingle_kernel(uint2 * outPairs, 
                                uint * cacheWriteLocation,
                                uint * writeStart,
								uint * overlappingCounts,
								Aabb * boxes, 
								uint maxBoxInd,
								int2 * internalNodeChildIndices, 
								Aabb * internalNodeAabbs, 
								Aabb * leafAabbs,
								KeyValuePair * mortonCodesAndAabbIndices,
								uint queryIdx, 
								uint treeIdx)
{
    uint boxIndex = blockIdx.x*blockDim.x + threadIdx.x;
    boxIndex = boxIndex<<NumSkip;
	if(boxIndex >= maxBoxInd) return;
	
	uint cacheSize = overlappingCounts[boxIndex];
	if(cacheSize < 1) return;
	
	uint writeLoc = cacheWriteLocation[boxIndex];
	//if(writeLoc - writeStart[boxIndex] >= cacheSize) return;
	
	const Aabb box = boxes[boxIndex];
	
	int stack[BVH_TRAVERSE_MAX_STACK_SIZE];
	int stackSize = 1;
	stack[0] = 0x80000000;
		
	int isLeaf;
    int iNode;
    int2 child;
    Aabb internalBox;
	
	for(;;) {
		if(outOfStack(stackSize)) break;
        
		iNode = stack[ stackSize - 1 ];
		stackSize--;
		
		iNode = getIndexWithInternalNodeMarkerRemoved(iNode);
		child = internalNodeChildIndices[iNode];
        isLeaf = isLeafNode(child);
        
		internalBox = internalNodeAabbs[iNode];
        
		if(isAabbOverlapping(box, internalBox))
		{
			if(isLeaf) {
			    writeOverlappings(outPairs,
                                writeLoc,
                                queryIdx,
                                boxIndex,
                                box,
                                treeIdx,
                                child,
                                mortonCodesAndAabbIndices,
                                leafAabbs);
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
#endif        //  #ifndef OVERLAPPING1_CUH

