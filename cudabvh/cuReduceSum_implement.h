#ifndef CUREDUCESUM_IMPLEMENT_H
#define CUREDUCESUM_IMPLEMENT_H

#include "reduce_common.h"

extern "C" {
void cuReduce_F_Sum(float * dst, float * src,
                    uint n, uint nBlocks, uint nThreads); 
}
#endif        //  #ifndef CUREDUCE_IMPLEMENT_H

