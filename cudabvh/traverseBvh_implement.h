#ifndef TRAVERSEBVH_IMPLEMENT_H
#define TRAVERSEBVH_IMPLEMENT_H

#include <bvh_common.h>
#include <radixsort_implement.h>

extern "C" void bvhRayTraverseIterative(RayInfo * rays,
								int * rootNodeIndex, 
								int2 * internalNodeChildIndex, 
								Aabb * internalNodeAabbs, 
								Aabb * leafNodeAabbs,
								KeyValuePair * mortonCodesAndAabbIndices,								
								uint numRays);

extern "C" void bvhTestRay(RayInfo * o_rays, float3 origin, float boxSize, uint w, uint n);

#endif        //  #ifndef TRAVERSEBVH_IMPLEMENT_H

