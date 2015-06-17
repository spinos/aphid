#include "Overlapping.cuh"
#include "Overlapping2.cuh"
#include "TetrahedronSystemInterface.h"

namespace bvhoverlap {

void writeLocation(uint * dst, uint * src, uint n)
{
    int tpb = 512;
    dim3 block(tpb, 1, 1);
    unsigned nblk = iDivUp(n, tpb);
    
    dim3 grid(nblk, 1, 1);
    
    startAsWriteLocation_kernel<<< grid, block >>>(dst, src, n);
}

void countPairsSelfCollideExclS(uint * dst, 
                                Aabb * boxes, 
                                uint numQueryInternalNodes,
								int2 * internalNodeChildIndex, 
								Aabb * internalNodeAabbs, 
								Aabb * leafNodeAabbs,
								KeyValuePair * mortonCodesAndAabbIndices,
								int * exclusionIndices)
{
    int nThreads = 16;
	dim3 block(nThreads, nThreads, 1);
    int nblk = numQueryInternalNodes;
    dim3 grid(nblk, 1, 1);

	countPairsSelfCollide_kernel<TETRAHEDRONSYSTEM_VICINITY_LENGTH, 16> <<< grid, block, 16320 >>>(dst,
                                boxes,
                                internalNodeChildIndex, 
								internalNodeAabbs, 
								leafNodeAabbs,
								mortonCodesAndAabbIndices,
								exclusionIndices);
}

void writePairCacheSelfCollideExclS(uint2 * dst, 
                                uint * locations, 
                                Aabb * boxes, 
                                uint numQueryInternalNodes,
								int2 * internalNodeChildIndex, 
								Aabb * internalNodeAabbs, 
								Aabb * leafNodeAabbs,
								KeyValuePair * mortonCodesAndAabbIndices,
								unsigned queryIdx,
								int * exclusionIndices)
{
    int nThreads = 16;
	dim3 block(nThreads, nThreads, 1);
    int nblk = numQueryInternalNodes;
    dim3 grid(nblk, 1, 1);
	
    writePairCacheSelfCollide_kernel<TETRAHEDRONSYSTEM_VICINITY_LENGTH, 16> <<< grid, block, 16320 >>>(dst, 
                                locations,
                                boxes,
                                internalNodeChildIndex, 
								internalNodeAabbs, 
								leafNodeAabbs,
								mortonCodesAndAabbIndices,
								queryIdx,
								exclusionIndices);
}

void countPairs(uint * dst,
                                Aabb * boxes,
                                KeyValuePair * queryIndirection,
                                uint numBoxes,
								int2 * internalNodeChildIndex, 
								Aabb * internalNodeAabbs, 
								Aabb * leafNodeAabbs,
								KeyValuePair * mortonCodesAndAabbIndices)
{ 
    int tpb = 64;
    dim3 block(tpb, 1, 1);
    unsigned nblk = iDivUp(numBoxes, tpb);
    
    dim3 grid(nblk, 1, 1);
    
    countPairs_kernel<64> <<< grid, block, 16320 >>>(dst,
                                boxes,
                                queryIndirection,
                                numBoxes,
								internalNodeChildIndex, 
								internalNodeAabbs, 
								leafNodeAabbs,
								mortonCodesAndAabbIndices);
}

void writePairCache(uint2 * dst, uint * locations, 
                                Aabb * boxes, 
                              KeyValuePair * queryIndirection,
                                uint numBoxes,
								int2 * internalNodeChildIndex, 
								Aabb * internalNodeAabbs, 
								Aabb * leafNodeAabbs,
								KeyValuePair * mortonCodesAndAabbIndices,
								unsigned queryIdx, unsigned treeIdx)
{
    int tpb = 64;
    dim3 block(tpb, 1, 1);
    unsigned nblk = iDivUp(numBoxes, tpb);
    
    dim3 grid(nblk, 1, 1);
    
    writePairCache_kernel<64> <<< grid, block, 16320 >>>(dst, 
                                locations,
                                boxes,
                                queryIndirection,
                                numBoxes,
								internalNodeChildIndex, 
								internalNodeAabbs, 
								leafNodeAabbs,
								mortonCodesAndAabbIndices,
								queryIdx, treeIdx);
}
}
