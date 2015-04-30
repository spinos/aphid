#ifndef CUREDUCESUM_IMPLEMENT_H
#define CUREDUCESUM_IMPLEMENT_H

#include "reduce_common.h"

template <class T>
void cuReduceFindSum(T *dst, T *src, 
    uint n, uint nBlocks, uint nThreads);

#endif        //  #ifndef CUREDUCE_IMPLEMENT_H

