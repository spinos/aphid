#ifndef CUREDUCEMINMAX_IMPLEMENT_H
#define CUREDUCEMINMAX_IMPLEMENT_H


#include <bvh_common.h>
    
template <class T>
void cuReduceFindMinMax(T *dst, T *src, unsigned numElements, unsigned numBlocks, unsigned numThreads);

template <class T, class T1>
void cuReduceFindMinMax(T *dst, T1 *src, unsigned numElements, unsigned numBlocks, unsigned numThreads);

#endif        //  #ifndef CUREDUCEMINMAX_IMPLEMENT_H

