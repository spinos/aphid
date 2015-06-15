#ifndef CUSMEM_CUH
#define CUSMEM_CUH

#include <cuda_runtime_api.h>

template<class T>
struct SharedMemory
{
    __device__ inline operator       T*()
    {
        extern __shared__ int __smem[];
        return (T*)__smem;
    }
};

#endif        //  #ifndef CUSMEM_CUH

