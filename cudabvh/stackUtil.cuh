#ifndef STACKUTIL_CUH
#define STACKUTIL_CUH

#include "bvh_math.cuh"
#include "radixsort_implement.h"
#include "Aabb.cuh"

#define B3_BROADPHASE_MAX_STACK_SIZE 64
#define B3_BROADPHASE_MAX_STACK_SIZE_M_2 62

inline __device__ int isStackFull(int stackSize)
{return stackSize > B3_BROADPHASE_MAX_STACK_SIZE_M_2; }

inline __device__ int outOfStack(int stackSize)
{return (stackSize < 1 || stackSize > B3_BROADPHASE_MAX_STACK_SIZE); }

#endif        //  #ifndef STACKUTIL_CUH

