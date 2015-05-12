#ifndef QUICKSORT_CUH
#define QUICKSORT_CUH

#include <cuda_runtime_api.h>
#include "SimpleQueue.cuh"

__global__ void quickSort_checkQ_kernel(uint * maxN,
                        simpleQueue::SimpleQueue * q)
{
   if(blockIdx.x < 1 && threadIdx.x < 1)
        maxN[0] = q->maxNumWorks();
}

__global__ void quickSort_kernel(int * obin,
                    int * idata,
                    int h,
                    int n)
{
    unsigned ind = blockIdx.x*blockDim.x + threadIdx.x;
	if(ind >= n) return;
	
	int d = idata[ind];
	
    atomicAdd(&obin[d/h], 1);
}

#endif        //  #ifndef QUICKSORT_CUH

