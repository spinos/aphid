#include "bvh_common.h"
#include "radixsort_implement.h"

namespace bvhoverlap {
void countPairs(uint * dst,
                                Aabb * boxes,
                                KeyValuePair * queryIndirection,
                                uint numBoxes,
								int2 * internalNodeChildIndex, 
								Aabb * internalNodeAabbs, 
								Aabb * leafNodeAabbs,
								KeyValuePair * mortonCodesAndAabbIndices);

void writePairCache(uint2 * dst, 
                                uint * locations, 
                               uint * cacheStarts, 
								uint * overlappingCounts,
								Aabb * boxes, 
                              KeyValuePair * queryIndirection,
                              uint numBoxes,
								int2 * internalNodeChildIndex, 
								Aabb * internalNodeAabbs, 
								Aabb * leafNodeAabbs,
								KeyValuePair * mortonCodesAndAabbIndices,
								unsigned queryIdx, unsigned treeIdx);

void countPairsSelfCollide(uint * dst, 
                                Aabb * boxes,
                                uint numQueryInternalNodes,
								uint numQueryPrimitives,
								int2 * internalNodeChildIndex, 
								Aabb * internalNodeAabbs, 
								Aabb * leafNodeAabbs,
								KeyValuePair * mortonCodesAndAabbIndices,
								int * exclusionIndices);

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
								int * exclusionIndices);
								
void writeLocation(uint * dst, uint * src, uint n);
}
