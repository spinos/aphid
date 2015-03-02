/*
 *  broadphase_implement.h
 *  cudabvh
 *
 *  Created by jian zhang on 2/20/15.
 *  Copyright 2015 __MyCompanyName__. All rights reserved.
 *
 */

#include "bvh_common.h"
#include <radixsort_implement.h>

extern "C" {

void broadphaseResetPairCounts(uint * dst, uint num);
void broadphaseResetPairCache(uint2 * dst, uint num);

void broadphaseComputePairCounts(uint * dst, Aabb * boxes, uint numBoxes,
								int * rootNodeIndex, 
								int2 * internalNodeChildIndex, 
								Aabb * internalNodeAabbs, 
								Aabb * leafNodeAabbs,
								KeyValuePair * mortonCodesAndAabbIndices,
								int isSelfCollision);

void broadphaseWritePairCache(uint2 * dst, uint * starts, uint * counts,
                              Aabb * boxes, uint numBoxes,
								int * rootNodeIndex, 
								int2 * internalNodeChildIndex, 
								Aabb * internalNodeAabbs, 
								Aabb * leafNodeAabbs,
								KeyValuePair * mortonCodesAndAabbIndices,
								unsigned queryIdx, unsigned treeIdx);

void broadphaseUniquePair(uint * dst, uint2 * pairs, uint pairLength, uint bufLength);
void broadphaseCompactUniquePairs(uint2 * dst, uint2 * pairs, uint * unique, uint * loc, uint pairLength);
}