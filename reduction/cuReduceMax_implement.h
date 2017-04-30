#ifndef CUREDUCEMAX_IMPLEMENT_H
#define CUREDUCEMAX_IMPLEMENT_H

#include <bvh_common.h>
    
template <class T>
void cuReduceFindMax(T *dst, T *src, unsigned numElements, unsigned numBlocks, unsigned numThreads);


#endif        //  #ifndef CUREDUCEMAX_IMPLEMENT_H

