#include "broadphase_implement.h"
#include <bvh_math.cu>
#include <CudaBase.h>

#define B3_BROADPHASE_MAX_STACK_SIZE 128

__global__ void resetPairCounts_kernel(uint * dst, uint maxInd)
{
    unsigned ind = blockIdx.x*blockDim.x + threadIdx.x;

	if(ind >= maxInd) return;

	dst[ind] = 0;
}

__global__ void computePairCounts_kernel(uint * overlappingCounts, Aabb * boxes,
                                uint maxBoxInd,
								int * rootNodeIndex, 
								int2 * internalNodeChildIndices, 
								Aabb * internalNodeAabbs, Aabb * leafAabbs,
								KeyValuePair * mortonCodesAndAabbIndices,
								int isSelfCollision)
{
	uint boxIndex = blockIdx.x*blockDim.x + threadIdx.x;
	if(boxIndex >= maxBoxInd) return;
	
	Aabb box = boxes[boxIndex];
	
	uint stack[B3_BROADPHASE_MAX_STACK_SIZE];
	
	int stackSize = 1;
	stack[0] = *rootNodeIndex;
		
	int isLeaf;
	
	while(stackSize > 0)
	{
		uint internalOrLeafNodeIndex = stack[ stackSize - 1 ];
		stackSize--;
		
		isLeaf = isLeafNode(internalOrLeafNodeIndex);	//Internal node if false
		uint bvhNodeIndex = getIndexWithInternalNodeMarkerRemoved(internalOrLeafNodeIndex);

		//bvhRigidIndex is not used if internal node
		int bvhRigidIndex = (isLeaf) ? mortonCodesAndAabbIndices[bvhNodeIndex].value : -1;
		
		Aabb bvhNodeAabb = (isLeaf) ? leafAabbs[bvhRigidIndex] : internalNodeAabbs[bvhNodeIndex];

		if(isAabbOverlapping(box, bvhNodeAabb))
		{    		    
			if(isLeaf)
			{
			    if(isSelfCollision) { // todo: connected elements shared same vertices
			        if(bvhRigidIndex != boxIndex) {
			            overlappingCounts[boxIndex] += 1;
			        }
			    } else {
			        overlappingCounts[boxIndex] += 1;
			    }
			}
			else {
                stack[ stackSize ] = internalNodeChildIndices[bvhNodeIndex].x;
                stackSize++;
                stack[ stackSize ] = internalNodeChildIndices[bvhNodeIndex].y;
                stackSize++;
			}
		}
		
	}
}

__global__ void writePairCache_kernel(uint2 * dst, uint * cacheStarts, uint * overlappingCounts, Aabb * boxes,
                                uint maxBoxInd,
								int * rootNodeIndex, 
								int2 * internalNodeChildIndices, 
								Aabb * internalNodeAabbs, Aabb * leafAabbs,
								KeyValuePair * mortonCodesAndAabbIndices,
								int isSelfCollision,
								unsigned queryIdx, unsigned treeIdx)
{
	uint boxIndex = blockIdx.x*blockDim.x + threadIdx.x;
	if(boxIndex >= maxBoxInd) return;
	
	//uint cacheSize = overlappingCounts[boxIndex];
	//if(cacheSize < 1) return;
	
	uint startLoc = cacheStarts[boxIndex];
	uint writeLoc = startLoc;
	
	Aabb box = boxes[boxIndex];
	
	uint stack[B3_BROADPHASE_MAX_STACK_SIZE];
	
	int stackSize = 1;
	stack[0] = *rootNodeIndex;
		
	int isLeaf;
	
	while(stackSize > 0)
	{
		uint internalOrLeafNodeIndex = stack[ stackSize - 1 ];
		stackSize--;
		
		isLeaf = isLeafNode(internalOrLeafNodeIndex);	//Internal node if false
		uint bvhNodeIndex = getIndexWithInternalNodeMarkerRemoved(internalOrLeafNodeIndex);

		//bvhRigidIndex is not used if internal node
		int bvhRigidIndex = (isLeaf) ? mortonCodesAndAabbIndices[bvhNodeIndex].value : -1;
		
		Aabb bvhNodeAabb = (isLeaf) ? leafAabbs[bvhRigidIndex] : internalNodeAabbs[bvhNodeIndex];

		if(isAabbOverlapping(box, bvhNodeAabb))
		{
			if(isLeaf)
			{
			    if(isSelfCollision) { // todo: connected elements shared same vertices
			        if(bvhRigidIndex != boxIndex) {
			            dst[writeLoc].x = combineObjectElementInd(queryIdx, boxIndex);
			            dst[writeLoc].y = combineObjectElementInd(treeIdx, bvhRigidIndex);
			            writeLoc++;
			        }
			    } else {
			        dst[writeLoc].x = combineObjectElementInd(queryIdx, boxIndex);
			        dst[writeLoc].y = combineObjectElementInd(treeIdx, bvhRigidIndex);
                    writeLoc++;
			    }
			    
			    //if((writeLoc - startLoc)==cacheSize) { // cache if full
			    //    return;
			    //}
			}
			else {
                stack[ stackSize ] = internalNodeChildIndices[bvhNodeIndex].x;
                stackSize++;
                stack[ stackSize ] = internalNodeChildIndices[bvhNodeIndex].y;
                stackSize++;
			}
		}
	}
}

extern "C" {

void broadphaseResetPairCounts(uint * dst, uint num)
{
    dim3 block(512, 1, 1);
    unsigned nblk = iDivUp(num, 512);
    dim3 grid(nblk, 1, 1);
    resetPairCounts_kernel<<< grid, block >>>(dst, num);
}

void broadphaseComputePairCounts(uint * dst,
                                Aabb * boxes,
                                uint numBoxes,
								int * rootNodeIndex, 
								int2 * internalNodeChildIndex, 
								Aabb * internalNodeAabbs, 
								Aabb * leafNodeAabbs,
								KeyValuePair * mortonCodesAndAabbIndices,
								int isSelfCollision)
{ 
    dim3 block(512, 1, 1);
    unsigned nblk = iDivUp(numBoxes, 512);
    
    dim3 grid(nblk, 1, 1);
    
    computePairCounts_kernel<<< grid, block >>>(dst,
                                boxes,
                                numBoxes,
								rootNodeIndex, 
								internalNodeChildIndex, 
								internalNodeAabbs, 
								leafNodeAabbs,
								mortonCodesAndAabbIndices,
								isSelfCollision);
}

void broadphaseWritePairCache(uint2 * dst, uint * starts, uint * counts,
                              Aabb * boxes, uint numBoxes,
								int * rootNodeIndex, 
								int2 * internalNodeChildIndex, 
								Aabb * internalNodeAabbs, 
								Aabb * leafNodeAabbs,
								KeyValuePair * mortonCodesAndAabbIndices,
								unsigned queryIdx, unsigned treeIdx)
{
    int tpb = CudaBase::LimitNThreadPerBlock(17, 50);
    dim3 block(tpb, 1, 1);
    unsigned nblk = iDivUp(numBoxes, tpb);
    
    dim3 grid(nblk, 1, 1);
    
    int isSelfCollision = (queryIdx == treeIdx);
    writePairCache_kernel<<< grid, block >>>(dst, starts, counts,
                                boxes,
                                numBoxes,
								rootNodeIndex, 
								internalNodeChildIndex, 
								internalNodeAabbs, 
								leafNodeAabbs,
								mortonCodesAndAabbIndices,
								isSelfCollision, queryIdx, treeIdx);
}

}

