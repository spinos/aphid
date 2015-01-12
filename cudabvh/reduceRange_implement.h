#ifndef REDUCERANGE_IMPLEMENT_H
#define REDUCERANGE_IMPLEMENT_H
#include <bvh_common.h>

extern "C" void bvhReduceFindMax(int *dst, int *src, unsigned numElements, unsigned numBlocks, unsigned numThreads);

#endif        //  #ifndef REDUCERANGE_IMPLEMENT_H

