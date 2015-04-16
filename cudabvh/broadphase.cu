#include "broadphase_implement.h"
#include <bvh_math.cu>
#include <CudaBase.h>
#include <stripedModel.cu>

#define B3_BROADPHASE_MAX_STACK_SIZE 128

__device__ int isTetrahedronConnected(uint a, uint b, const uint4 * v)
{
    if(a == b) return 1;
    
    const uint4 ta = v[a];
    const uint4 tb = v[b];
    
    if(ta.x == tb.x || ta.x == tb.y || ta.x == tb.z || ta.x == tb.w) return 1;
    if(ta.y == tb.x || ta.y == tb.y || ta.y == tb.z || ta.y == tb.w) return 1;
    if(ta.z == tb.x || ta.z == tb.y || ta.z == tb.z || ta.z == tb.w) return 1;
    if(ta.w == tb.x || ta.w == tb.y || ta.w == tb.z || ta.w == tb.w) return 1;
    
    return 0;
}

__global__ void resetPairCounts_kernel(uint * dst, uint maxInd)
{
    unsigned ind = blockIdx.x*blockDim.x + threadIdx.x;

	if(ind >= maxInd) return;

	dst[ind] = 0;
}

__global__ void resetPairCache_kernel(uint2 * dst, uint maxInd)
{
    unsigned ind = blockIdx.x*blockDim.x + threadIdx.x;

	if(ind >= maxInd) return;

	dst[ind].x = 0x80000000;
	dst[ind].y = 0x80000000;
}

__global__ void computePairCounts_kernel(uint * overlappingCounts, Aabb * boxes,
                                uint maxBoxInd,
								int * rootNodeIndex, 
								int2 * internalNodeChildIndices, 
								Aabb * internalNodeAabbs, Aabb * leafAabbs,
								KeyValuePair * mortonCodesAndAabbIndices)
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
			    overlappingCounts[boxIndex] += 1;
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

__global__ void computePairCountsSelfCollide_kernel(uint * overlappingCounts, Aabb * boxes,
                                uint maxBoxInd,
								int * rootNodeIndex, 
								int2 * internalNodeChildIndices, 
								Aabb * internalNodeAabbs, Aabb * leafAabbs,
								KeyValuePair * mortonCodesAndAabbIndices,
								uint4 * tetrahedronIndices)
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
			    if(!isTetrahedronConnected(bvhRigidIndex, boxIndex, tetrahedronIndices)) 
			        overlappingCounts[boxIndex] += 1;
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
		uint2 pair;
		if(isAabbOverlapping(box, bvhNodeAabb))
		{
			if(isLeaf)
			{
			    pair.x = combineObjectElementInd(queryIdx, boxIndex);
                pair.y = combineObjectElementInd(treeIdx, bvhRigidIndex);
                ascentOrder<uint2>(pair);
                dst[writeLoc] = pair;
                writeLoc++;
			    // }
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

__global__ void writePairCacheSelfCollide_kernel(uint2 * dst, uint * cacheStarts, uint * overlappingCounts, Aabb * boxes,
                                uint maxBoxInd,
								int * rootNodeIndex, 
								int2 * internalNodeChildIndices, 
								Aabb * internalNodeAabbs, Aabb * leafAabbs,
								KeyValuePair * mortonCodesAndAabbIndices,
								uint4 * tetrahedronIndices,
								unsigned queryIdx)
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
		uint2 pair;
		if(isAabbOverlapping(box, bvhNodeAabb))
		{
			if(isLeaf)
			{
			    if(!isTetrahedronConnected(bvhRigidIndex, boxIndex, tetrahedronIndices)) {
			        pair.x = combineObjectElementInd(queryIdx, boxIndex);
			        pair.y = combineObjectElementInd(queryIdx, bvhRigidIndex);
			        ascentOrder<uint2>(pair);
			        dst[writeLoc] = pair;
			        writeLoc++;
			    }
			    // }
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

__global__ void uniquePair_kernel(uint * dst, uint2 * pairs, uint pairLength, uint maxInd)
{
    unsigned ind = blockIdx.x*blockDim.x + threadIdx.x;

	if(ind >= maxInd) return;
	
	if(ind >= pairLength) {
	    dst[ind] = 0;
	    return;
	}
	
// assume it is unique
	dst[ind] = 1;
	
	uint a = pairs[ind].x;
	uint b = pairs[ind].y;

	unsigned cur = ind;
// check forward
	for(;;) {
	    if(cur < 1) return;
	    cur--;
	    if(pairs[cur].x != a) return;
	    if(pairs[cur].y == b) {
	        dst[ind] = 0;
	        return;
	    }
	}
}

__global__ void compactUniquePairs_kernel(uint2 * dst, uint2 * pairs, uint * unique, uint * dstLoc, uint maxInd)
{
    unsigned ind = blockIdx.x*blockDim.x + threadIdx.x;

	if(ind >= maxInd) return;
	
	if(unique[ind] > 0) {
	    dst[dstLoc[ind]] = pairs[ind];
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

void broadphaseResetPairCache(uint2 * dst, uint num)
{
    dim3 block(512, 1, 1);
    unsigned nblk = iDivUp(num, 512);
    dim3 grid(nblk, 1, 1);
    resetPairCache_kernel<<< grid, block >>>(dst, num);
}

void broadphaseComputePairCounts(uint * dst,
                                Aabb * boxes,
                                uint numBoxes,
								int * rootNodeIndex, 
								int2 * internalNodeChildIndex, 
								Aabb * internalNodeAabbs, 
								Aabb * leafNodeAabbs,
								KeyValuePair * mortonCodesAndAabbIndices)
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
								mortonCodesAndAabbIndices);
}

void broadphaseComputePairCountsSelfCollide(uint * dst, Aabb * boxes, uint numBoxes,
								int * rootNodeIndex, 
								int2 * internalNodeChildIndex, 
								Aabb * internalNodeAabbs, 
								Aabb * leafNodeAabbs,
								KeyValuePair * mortonCodesAndAabbIndices,
								uint4 * tetrahedronIndices)
{
    int tpb = CudaBase::LimitNThreadPerBlock(16, 50);
    dim3 block(tpb, 1, 1);
    unsigned nblk = iDivUp(numBoxes, tpb);
    
    dim3 grid(nblk, 1, 1);
    
    computePairCountsSelfCollide_kernel<<< grid, block >>>(dst,
                                boxes,
                                numBoxes,
								rootNodeIndex, 
								internalNodeChildIndex, 
								internalNodeAabbs, 
								leafNodeAabbs,
								mortonCodesAndAabbIndices,
								tetrahedronIndices);
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
    int tpb = CudaBase::LimitNThreadPerBlock(18, 50);
    dim3 block(tpb, 1, 1);
    unsigned nblk = iDivUp(numBoxes, tpb);
    
    dim3 grid(nblk, 1, 1);
    
    writePairCache_kernel<<< grid, block >>>(dst, starts, counts,
                                boxes,
                                numBoxes,
								rootNodeIndex, 
								internalNodeChildIndex, 
								internalNodeAabbs, 
								leafNodeAabbs,
								mortonCodesAndAabbIndices,
								queryIdx, treeIdx);
}

void broadphaseWritePairCacheSelfCollide(uint2 * dst, uint * starts, uint * counts,
                              Aabb * boxes, uint numBoxes,
								int * rootNodeIndex, 
								int2 * internalNodeChildIndex, 
								Aabb * internalNodeAabbs, 
								Aabb * leafNodeAabbs,
								KeyValuePair * mortonCodesAndAabbIndices,
								uint4 * tetrahedronIndices,
								unsigned queryIdx)
{
    int tpb = CudaBase::LimitNThreadPerBlock(20, 50);
    dim3 block(tpb, 1, 1);
    unsigned nblk = iDivUp(numBoxes, tpb);
    
    dim3 grid(nblk, 1, 1);
    
    writePairCacheSelfCollide_kernel<<< grid, block >>>(dst, starts, counts,
                                boxes,
                                numBoxes,
								rootNodeIndex, 
								internalNodeChildIndex, 
								internalNodeAabbs, 
								leafNodeAabbs,
								mortonCodesAndAabbIndices,
								tetrahedronIndices,
								queryIdx);
}

void broadphaseUniquePair(uint * dst, uint2 * pairs, uint pairLength, uint bufLength)
{
    dim3 block(512, 1, 1);
    unsigned nblk = iDivUp(bufLength, 512);
    dim3 grid(nblk, 1, 1);
    uniquePair_kernel<<< grid, block >>>(dst, pairs, pairLength, bufLength);
}

void broadphaseCompactUniquePairs(uint2 * dst, uint2 * pairs, uint * unique, uint * loc, uint pairLength)
{
    dim3 block(512, 1, 1);
    unsigned nblk = iDivUp(pairLength, 512);
    dim3 grid(nblk, 1, 1);
    
    compactUniquePairs_kernel<<< grid, block >>>(dst, pairs, unique, loc, pairLength);
}

}

