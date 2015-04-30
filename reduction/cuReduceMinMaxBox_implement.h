#ifndef CUREDUCEMINMAXBOX_IMPLEMENT_H
#define CUREDUCEMINMAXBOX_IMPLEMENT_H

#include <bvh_common.h>
    
template <class T>
void cuReduceFindMinMaxBox(T *dst, T *src, unsigned numElements, unsigned numBlocks, unsigned numThreads);

template <class T, class T1>
void cuReduceFindMinMaxBox(T *dst, T1 *src, unsigned numElements, unsigned numBlocks, unsigned numThreads);

#endif        //  #ifndef CUREDUCEMINMAXBOX_IMPLEMENT_H

