#include "bvh_common.h"
#include "radixsort_implement.h"

namespace bvhoverlap {
void countPairs(uint * dst,
                                Aabb * boxes,
                                uint numBoxes,
								int2 * internalNodeChildIndex, 
								Aabb * internalNodeAabbs, 
								Aabb * leafNodeAabbs,
								KeyValuePair * mortonCodesAndAabbIndices);

void countPairsSelfCollideExclS(uint * dst, 
                                Aabb * boxes, 
                                uint numBoxes,
								int2 * internalNodeChildIndex, 
								Aabb * internalNodeAabbs, 
								Aabb * leafNodeAabbs,
								KeyValuePair * mortonCodesAndAabbIndices,
								int * exclusionIndices);

void writePairCache(uint2 * dst, uint * locations, 
                                uint * starts, uint * counts,
                              Aabb * boxes, uint numBoxes,
								int2 * internalNodeChildIndex, 
								Aabb * internalNodeAabbs, 
								Aabb * leafNodeAabbs,
								KeyValuePair * mortonCodesAndAabbIndices,
								unsigned queryIdx, unsigned treeIdx);
								
void writePairCacheSelfCollideExclS(uint2 * dst, 
                                uint * locations, 
								uint * starts, 
								uint * counts,
								Aabb * boxes, uint numBoxes,
								int * rootNodeIndex, 
								int2 * internalNodeChildIndex, 
								Aabb * internalNodeAabbs, 
								Aabb * leafNodeAabbs,
								KeyValuePair * mortonCodesAndAabbIndices,
								unsigned queryIdx,
								int * exclusionIndices);
								
void writeLocation(uint * dst, uint * src, uint n);
}
