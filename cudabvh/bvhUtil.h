#ifndef BVH_UTIL_H
#define BVH_UTIL_H

#include "radixsort_implement.h"
#include "cuSMem.cuh"
#include "cuReduceInBlock.cuh"
#include "bvh_math.cuh"
#include "Aabb.cuh"
#include "stripedModel.cu"

#define BVH_PACKET_TRAVERSE_CACHE_SIZE 64
#define BVH_TRAVERSE_MAX_STACK_SIZE 64

inline __device__ uint tId1()
{ return threadIdx.x; }

inline __device__ uint tId2()
{ return blockDim.x * threadIdx.y + threadIdx.x; }

inline __device__ int isInternalNode(const int2 & child)
{ return (child.x>>31) != 0; }

inline __device__ int isLeafNode(const int2 & child)
{ return (child.x>>31) == 0; }

template<int NumThreads>
inline __device__ void putLeafBoxAndIndInSmem(Aabb * elementBox,
                                    uint * elementInd,
                                   uint tid,
                                   uint n,
                                   int2 range,
                                   KeyValuePair * elementHash,
                                   Aabb * leafAabbs)
{
    uint iElement; 
    uint loc = tid;
    if(loc < n) {
        iElement = elementHash[range.x + loc].value;
        elementBox[loc] = leafAabbs[iElement];
        elementInd[loc] = iElement;
    }
}

template<int NumExcls>
__device__ void writeElementExclusion(int * dst,
									int * exclusionInd)
{
    int i=0;
#if 1
    int4 * dstInd4 = (int4 *)dst;
	int4 * srcInd4 = (int4 *)exclusionInd;
	for(;i<(NumExcls>>2); i++)
	    dstInd4[i] = srcInd4[i];
#else
    for(;i<NumExcls; i++)
	    dst[i] = exclusionInd[i];
#endif
}

template<int NumExcls>
__device__ int isElementExcludedS(int b, int * exclusionInd)
{
	int i;
#if 1
    int4 * exclusionInd4 = (int4 *)exclusionInd;
	for(i=0; i<(NumExcls>>2); i++) {
		if(exclusionInd4[i].x < 0) break;
		if(b <= exclusionInd4[i].x) return 1;
		if(exclusionInd4[i].y < 0) break;
		if(b <= exclusionInd4[i].y) return 1;
		if(exclusionInd4[i].z < 0) break;
		if(b <= exclusionInd4[i].z) return 1;
		if(exclusionInd4[i].w < 0) break;
		if(b <= exclusionInd4[i].w) return 1;
	}
#else
    for(i=0; i<NumExcls; i++) {
		if(exclusionInd[i] < 0) break;
		if(b <= exclusionInd[i]) return 1;
	}
#endif
	return 0;
}
#endif        //  #ifndef BVH_UTIL_H

