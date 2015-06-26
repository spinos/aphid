#include "Overlapping.cuh"
#include "Overlapping1.cuh"
#include "Overlapping2.cuh"
#include "Overlapping3.cuh"
#include "TetrahedronSystemInterface.h"
#define USE_PACKET_TRAVERSE 0
#define SINGL_TRAVERSE_NUM_THREAD 64
#define SINGL_TRAVERSE_SIZE_SMEM 16320
#define SELF_COLLIDE_SKIP 1
#define SOLLIDE_SKIP 1
//#define SINGL_TRAVERSE_NUM_THREAD 128
//#define SINGL_TRAVERSE_SIZE_SMEM 32640

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
                                uint * anchors,
                                int4 * tetrahedronVertices,
                                uint numQueryInternalNodes,
								uint numQueryPrimitives,
								int2 * internalNodeChildIndex, 
								Aabb * internalNodeAabbs, 
								Aabb * leafNodeAabbs,
								KeyValuePair * mortonCodesAndAabbIndices,
								int * exclusionIndices)
{
#if USE_PACKET_TRAVERSE
    int nThreads = 256;
	dim3 block(nThreads, 1, 1);
    int nblk = iDivUp(numQueryInternalNodes, 8);
    dim3 grid(nblk, 1, 1);

	countPairsSelfCollidePacket_kernel<TETRAHEDRONSYSTEM_VICINITY_LENGTH> <<< grid, block, 16320 >>>(dst,
                                boxes,
                                numQueryInternalNodes,
                                internalNodeChildIndex, 
								internalNodeAabbs, 
								leafNodeAabbs,
								mortonCodesAndAabbIndices,
								exclusionIndices);
#else
    int nThreads = SINGL_TRAVERSE_NUM_THREAD;
	dim3 block(nThreads, 1, 1);
    int nblk = iDivUp(numQueryPrimitives>>SELF_COLLIDE_SKIP, nThreads);
    dim3 grid(nblk, 1, 1);

	countPairsSelfCollideSingle_kernel<TETRAHEDRONSYSTEM_VICINITY_LENGTH, SELF_COLLIDE_SKIP> <<< grid, block, SINGL_TRAVERSE_SIZE_SMEM >>>(dst,
                                boxes,
                                anchors,
                                tetrahedronVertices,
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
    int nThreads = SINGL_TRAVERSE_NUM_THREAD;
	dim3 block(nThreads, 1, 1);
    int nblk = iDivUp(numQueryPrimitives>>SELF_COLLIDE_SKIP, nThreads);;
    dim3 grid(nblk, 1, 1);
	
    writePairCacheSelfCollideSingle_kernel<TETRAHEDRONSYSTEM_VICINITY_LENGTH, SELF_COLLIDE_SKIP> <<< grid, block, SINGL_TRAVERSE_SIZE_SMEM >>>(dst, 
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
                                uint * anchors,
                                int4 * tetrahedronVertices,
                                KeyValuePair * queryIndirection,
                                uint numQueryInternalNodes,
                                uint numBoxes,
								int2 * internalNodeChildIndex, 
								Aabb * internalNodeAabbs, 
								Aabb * leafNodeAabbs,
								KeyValuePair * mortonCodesAndAabbIndices)
{ 
#if USE_PACKET_TRAVERSE
    int nThreads = 16;
	dim3 block(nThreads, nThreads, 1);
    int nblk = numQueryInternalNodes;
    dim3 grid(nblk, 1, 1);
    
    countPairsPacket_kernel<16> <<< grid, block, 16320 >>>(dst,
                                boxes,
                                queryIndirection,
                                internalNodeChildIndex, 
								internalNodeAabbs, 
								leafNodeAabbs,
								mortonCodesAndAabbIndices);
#else
    int tpb = SINGL_TRAVERSE_NUM_THREAD;
    dim3 block(tpb, 1, 1);
    unsigned nblk = iDivUp(numBoxes>>SOLLIDE_SKIP, tpb);
    
    dim3 grid(nblk, 1, 1);
    
    countPairsSingle_kernel<SOLLIDE_SKIP> <<< grid, block >>>(dst,
                                boxes,
                                anchors,
                                tetrahedronVertices,
                                numBoxes,
								internalNodeChildIndex, 
								internalNodeAabbs, 
								leafNodeAabbs,
								mortonCodesAndAabbIndices);
#endif
}

void writePairCache(uint2 * dst, 
                                uint * locations, 
                                uint * cacheStarts, 
								uint * overlappingCounts,
								Aabb * boxes, 
                              KeyValuePair * queryIndirection,
                                uint numQueryInternalNodes,
                                uint numBoxes,
								int2 * internalNodeChildIndex, 
								Aabb * internalNodeAabbs, 
								Aabb * leafNodeAabbs,
								KeyValuePair * mortonCodesAndAabbIndices,
								uint queryIdx, 
								uint treeIdx)
{
#if USE_PACKET_TRAVERSE
    int nThreads = 16;
	dim3 block(nThreads, nThreads, 1);
    int nblk = numQueryInternalNodes;
    dim3 grid(nblk, 1, 1);
    
    writePairCachePacket_kernel<16> <<< grid, block, 16320 >>>(dst, 
                                locations,
                                boxes,
                                queryIndirection,
                                numBoxes,
								internalNodeChildIndex, 
								internalNodeAabbs, 
								leafNodeAabbs,
								mortonCodesAndAabbIndices,
								queryIdx, 
								treeIdx);
#else
    int tpb = SINGL_TRAVERSE_NUM_THREAD;
    dim3 block(tpb, 1, 1);
    unsigned nblk = iDivUp(numBoxes>>SOLLIDE_SKIP, tpb);
    
    dim3 grid(nblk, 1, 1);
    
    writePairCacheSingle_kernel<SOLLIDE_SKIP> <<< grid, block >>>(dst,
                                locations,
                                cacheStarts,
                                overlappingCounts,
                                boxes,
                                numBoxes,
								internalNodeChildIndex, 
								internalNodeAabbs, 
								leafNodeAabbs,
								mortonCodesAndAabbIndices,
								queryIdx, 
								treeIdx);
#endif
}
}
