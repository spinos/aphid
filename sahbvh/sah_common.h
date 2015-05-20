#ifndef SAH_COMMON_H
#define SAH_COMMON_H

#include "radixsort_implement.h"
#include "bvh_common.h"

struct SplitBin {
    Aabb leftBox;
    uint leftCount;
    Aabb rightBox;
    uint rightCount;
    int id;
    float plane;
};

#define SIZE_OF_SPLITBIN 64

#endif        //  #ifndef SAH_COMMON_H

