#ifndef ALLBASE_H
#define ALLBASE_H

#include <cuda_runtime_api.h>

typedef unsigned int uint;
typedef unsigned long long uint64;

inline int getNumBlock(int x, int b)
{ return ( (x & (b-1)) == 0 ) ? (x / b) : (x / b + 1); }

#endif        //  #ifndef ALLBASE_H

