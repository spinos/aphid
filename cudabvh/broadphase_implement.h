/*
 *  broadphase_implement.h
 *  cudabvh
 *
 *  Created by jian zhang on 2/20/15.
 *  Copyright 2015 __MyCompanyName__. All rights reserved.
 *
 */

#include "bvh_common.h"
#include "radixsort_implement.h"

extern "C" {

void broadphaseResetPairCounts(uint * dst, uint num);
void broadphaseResetPairCache(uint2 * dst, uint num);

void broadphaseComputePairCounts(uint * dst, Aabb * boxes, uint numBoxes,
								int * rootNodeIndex, 
								int2 * internalNodeChildIndex, 
								Aabb * internalNodeAabbs, 
								Aabb * leafNodeAabbs,
								KeyValuePair * mortonCodesAndAabbIndices);

void broadphaseComputePairCountsSelfCollide(uint * dst, Aabb * boxes, uint numBoxes,
								int * rootNodeIndex, 
								int2 * internalNodeChildIndex, 
								Aabb * internalNodeAabbs, 
								Aabb * leafNodeAabbs,
								KeyValuePair * mortonCodesAndAabbIndices,
								uint4 * tetrahedronIndices);

void cuBroadphase_writePairCache(uint2 * dst, uint * locations, 
								uint * starts, uint * counts,
                              Aabb * boxes, uint numBoxes,
								int * rootNodeIndex, 
								int2 * internalNodeChildIndex, 
								Aabb * internalNodeAabbs, 
								Aabb * leafNodeAabbs,
								KeyValuePair * mortonCodesAndAabbIndices,
								unsigned queryIdx, unsigned treeIdx);

void broadphaseWritePairCacheSelfCollide(uint2 * dst, uint * starts, uint * counts,
                              Aabb * boxes, uint numBoxes,
								int * rootNodeIndex, 
								int2 * internalNodeChildIndex, 
								Aabb * internalNodeAabbs, 
								Aabb * leafNodeAabbs,
								KeyValuePair * mortonCodesAndAabbIndices,
								uint4 * tetrahedronIndices,
								unsigned queryIdx);

void broadphaseUniquePair(uint * dst, uint2 * pairs, uint pairLength, uint bufLength);
void broadphaseCompactUniquePairs(uint2 * dst, uint2 * pairs, uint * unique, uint * loc, uint pairLength);

void broadphaseComputePairCountsSelfCollideExclusion(uint * dst, Aabb * boxes, uint numBoxes,
								int * rootNodeIndex, 
								int2 * internalNodeChildIndex, 
								Aabb * internalNodeAabbs, 
								Aabb * leafNodeAabbs,
								KeyValuePair * mortonCodesAndAabbIndices,
								uint * exclusionIndices,
								uint * exclusionStarts);
								
void cuBroadphase_writePairCacheSelfCollideExclusion(uint2 * dst, uint * locations, 
								uint * starts, uint * counts,
                              Aabb * boxes, uint numBoxes,
								int * rootNodeIndex, 
								int2 * internalNodeChildIndex, 
								Aabb * internalNodeAabbs, 
								Aabb * leafNodeAabbs,
								KeyValuePair * mortonCodesAndAabbIndices,
								unsigned queryIdx,
								uint * exclusionIndices,
								uint * exclusionStarts);
}

namespace cubroadphase {
void countPairsSelfCollideExclS(uint * dst, Aabb * boxes, uint numBoxes,
								int * rootNodeIndex, 
								int2 * internalNodeChildIndex, 
								Aabb * internalNodeAabbs, 
								// int * internalChildLimit,
								Aabb * leafNodeAabbs,
								KeyValuePair * mortonCodesAndAabbIndices,
								uint * exclusionIndices,
								uint * exclusionStarts,
								int nThreads);
								
void writePairCacheSelfCollideExclS(uint2 * dst, uint * locations, 
								uint * starts, uint * counts,
                              Aabb * boxes, uint numBoxes,
								int * rootNodeIndex, 
								int2 * internalNodeChildIndex, 
								Aabb * internalNodeAabbs, 
								// int * internalChildLimit,
								Aabb * leafNodeAabbs,
								KeyValuePair * mortonCodesAndAabbIndices,
								unsigned queryIdx,
								uint * exclusionIndices,
								uint * exclusionStarts,
								int nThreads);
								
void writeLocation(uint * dst, uint * src, uint n);
}