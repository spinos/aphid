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

#endif        //  #ifndef BVH_UTIL_H

