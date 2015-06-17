#include "Overlapping.cuh"
#include "Overlapping2.cuh"
#include "Overlapping3.cuh"
#include "TetrahedronSystemInterface.h"
#define USE_PACKET_TRAVERSE 0

namespace bvhoverlap {

void writeLocation(uint * dst, uint * src, uint n)
{
    int tpb = 512;
    dim3 block(tpb, 1, 1);
    unsigned nblk = iDivUp(n, tpb);
    
    dim3 grid(nblk, 1, 1);
    
    startAsWriteLocation_kernel<<< grid, block >>>(dst, src, n);
}

void countPairsSelfCollide(uint * dst, 
                                Aabb * boxes, 
                                uint numQueryInternalNodes,
								uint numQueryPrimitives,
								int2 * internalNodeChildIndex, 
								Aabb * internalNodeAabbs, 
								Aabb * leafNodeAabbs,
								KeyValuePair * mortonCodesAndAabbIndices,
								int * exclusionIndices)
{
#if USE_PACKET_TRAVERSE
    int nThreads = 16;
	dim3 block(nThreads, nThreads, 1);
    int nblk = numQueryInternalNodes;
    dim3 grid(nblk, 1, 1);

	countPairsSelfCollidePacket_kernel<TETRAHEDRONSYSTEM_VICINITY_LENGTH, 16> <<< grid, block, 16320 >>>(dst,
                                boxes,
                                internalNodeChildIndex, 
								internalNodeAabbs, 
								leafNodeAabbs,
								mortonCodesAndAabbIndices,
								exclusionIndices);
#else
    int nThreads = 64;
	dim3 block(nThreads, 1, 1);
    int nblk = iDivUp(numQueryPrimitives, nThreads);
    dim3 grid(nblk, 1, 1);

	countPairsSelfCollideSingle_kernel<TETRAHEDRONSYSTEM_VICINITY_LENGTH, 64> <<< grid, block, 16320 >>>(dst,
                                boxes,
                                numQueryPrimitives,
                                internalNodeChildIndex, 
								internalNodeAabbs, 
								leafNodeAabbs,
								mortonCodesAndAabbIndices,
								exclusionIndices);
#endif
}

void writePairCacheSelfCollide(uint2 * dst, 
                                uint * locations, 
                                uint * cacheStarts, 
								uint * overlappingCounts,
                                Aabb * boxes, 
                                uint numQueryInternalNodes,
								uint numQueryPrimitives,
								int2 * internalNodeChildIndex, 
								Aabb * internalNodeAabbs, 
								Aabb * leafNodeAabbs,
								KeyValuePair * mortonCodesAndAabbIndices,
								unsigned queryIdx,
								int * exclusionIndices)
{
#if USE_PACKET_TRAVERSE
    int nThreads = 16;
	dim3 block(nThreads, nThreads, 1);
    int nblk = numQueryInternalNodes;
    dim3 grid(nblk, 1, 1);
	
    writePairCacheSelfCollidePacket_kernel<TETRAHEDRONSYSTEM_VICINITY_LENGTH, 16> <<< grid, block, 16320 >>>(dst, 
                                locations,
                                overlappingCounts,
                                boxes,
                                internalNodeChildIndex, 
								internalNodeAabbs, 
								leafNodeAabbs,
								mortonCodesAndAabbIndices,
								queryIdx,
								exclusionIndices);
#else
    int nThreads = 64;
	dim3 block(nThreads, 1, 1);
    int nblk = iDivUp(numQueryPrimitives, nThreads);;
    dim3 grid(nblk, 1, 1);
	
    writePairCacheSelfCollideSingle_kernel<TETRAHEDRONSYSTEM_VICINITY_LENGTH, 16> <<< grid, block, 16320 >>>(dst, 
                                locations,
                                overlappingCounts,
                                boxes,
                                numQueryPrimitives,
                                internalNodeChildIndex, 
								internalNodeAabbs, 
								leafNodeAabbs,
								mortonCodesAndAabbIndices,
								queryIdx,
								exclusionIndices);
#endif
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
