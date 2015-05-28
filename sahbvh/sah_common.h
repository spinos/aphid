#ifndef SAH_COMMON_H
#define SAH_COMMON_H

#include "radixsort_implement.h"
#include "bvh_common.h"

struct __align__(4) SplitBin {
    Aabb leftBox;
    uint leftCount;
    Aabb rightBox;
    uint rightCount;
    int dimension;
    float plane;
};

#define SIZE_OF_SPLITBIN 64
#define SIZE_OF_SPLITBIN_IN_INT 16

#endif        //  #ifndef SAH_COMMON_H

