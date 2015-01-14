#ifndef CREATEBVH_IMPLEMENT_H
#define CREATEBVH_IMPLEMENT_H

#include "bvh_common.h"
#include <radixsort_implement.h>

extern "C" void bvhCalculateLeafAabbsTriangle(Aabb *dst, float3 * cvs, uint3 * tris, unsigned numTriangles);

extern "C" void bvhCalculateLeafAabbs(Aabb *dst, float3 * cvs, EdgeContact * edges, unsigned numEdges, unsigned numVertices);

extern "C" void bvhCalculateLeafHash(KeyValuePair * dst, Aabb * leafBoxes, uint numLeaves, Aabb * bigBox);

extern "C" void bvhComputeAdjacentPairCommonPrefix(KeyValuePair * mortonCode,
													uint64 * o_commonPrefix,
													int * o_commonPrefixLength,
													uint numInternalNodes);
													
extern "C" void bvhConnectLeafNodesToInternalTree(int * commonPrefixLengths, 
								int * o_leafNodeParentIndex,
								int2 * o_internalNodeChildIndex, 
								uint numLeafNodes);
								
extern "C" void bvhConnectInternalTreeNodes(uint64 * commonPrefix, int * commonPrefixLengths,
											int2 * o_internalNodeChildIndex,
											int * o_internalNodeParentIndex,
											int * o_rootNodeIndex,
											uint numInternalNodes);
											
extern "C" void bvhFindDistanceFromRoot(int* rootNodeIndex, int* internalNodeParentNodes,
									int* out_distanceFromRoot, 
									uint numInternalNodes);

extern "C" void bvhFormInternalNodeAabbsAtDistance(int * distanceFromRoot, KeyValuePair * mortonCodesAndAabbIndices,
												int2 * childNodes,
												Aabb * leafNodeAabbs, Aabb * internalNodeAabbs,
												int maxDistanceFromRoot, int processedDistance, 
												uint numInternalNodes);
									
#endif        //  #ifndef CREATEBVH_IMPLEMENT_H

