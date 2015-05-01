#ifndef SCAN_IMPLEMENT_H
#define SCAN_IMPLEMENT_H

#include "bvh_common.h"

extern "C" void scanExclusive(
    uint *d_Dst,
    uint *d_Src,
    uint *d_intermediate,
    uint batchSize,
    uint arrayLength
);
#endif        //  #ifndef SCAN_IMPLEMENT_H

