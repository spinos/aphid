/*  reduction in block
 *  32 <= ntrhead <= 256
 */

#ifndef CUREDUCEINBLOCK_CUH
#define CUREDUCEINBLOCK_CUH

#include <cuda_runtime_api.h>

template<int NumThreads, typename T>
__device__ void reduceSumInBlock(int tid, T * m)
{
    if(NumThreads > 128) {
        if(tid < 128) m[tid] += m[tid + 128];
        __syncthreads();
    }
    
    if(NumThreads > 64) {
        if(tid < 64) m[tid] += m[tid + 64];
        __syncthreads();
    }
    
    if(NumThreads > 32) {
       if(tid < 32) m[tid] += m[tid + 32];
    }

    if(tid < 16) m[tid] += m[tid + 16];

    if(tid < 8) m[tid] += m[tid + 8];

    if(tid < 4) m[tid] += m[tid + 4];

    if(tid < 2) m[tid] += m[tid + 2];

    if(tid < 1) m[tid] += m[tid + 1];
}

template<int NumThreads, typename T>
__device__ void reduceMaxInBlock(int tid, T * m)
{
    if(tid < 128 && NumThreads >= 256) {
        m[tid] = max(m[tid], m[tid + 128]);
        __syncthreads();
    }
    
    if(tid < 64 && NumThreads >= 128) {
        m[tid] = max(m[tid], m[tid + 64]);
        __syncthreads();
    }
    
    if(tid < 32 && NumThreads >= 64) {
        m[tid] = max(m[tid], m[tid + 32]);
        //__syncthreads();
    }
    
    if(tid < 16 && NumThreads >= 32) {
        m[tid] = max(m[tid], m[tid + 16]);
    }
    //__syncthreads();
    
    if(tid < 8 && NumThreads >= 16) {
        m[tid] = max(m[tid], m[tid + 8]);
    }
    //__syncthreads();
    
    if(tid < 4 && NumThreads >= 8) {
        m[tid] = max(m[tid], m[tid + 4]);
    }
    //__syncthreads();
    
    if(tid < 2 && NumThreads >= 4) {
        m[tid] = max(m[tid], m[tid + 2]);
    }
    //__syncthreads();
    
    if(tid < 1) {
        m[tid] = max(m[tid], m[tid + 1]);
    }
    //__syncthreads();
}

template<int NumThreads, typename T>
__device__ void reduceMinInBlock(int tid, T * m)
{
    if(tid < 128 && NumThreads >= 256) {
        m[tid] = min(m[tid], m[tid + 128]);
        __syncthreads();
    }
    
    if(tid < 64 && NumThreads >= 128) {
        m[tid] = min(m[tid], m[tid + 64]);
        __syncthreads();
    }
    
    if(tid < 32 && NumThreads >= 64) {
        m[tid] = min(m[tid], m[tid + 32]);
        //__syncthreads();
    }
    
    if(tid < 16 && NumThreads >= 32) {
        m[tid] = min(m[tid], m[tid + 16]);
    }
    //__syncthreads();
    
    if(tid < 8 && NumThreads >= 16) {
        m[tid] = min(m[tid], m[tid + 8]);
    }
    //__syncthreads();
    
    if(tid < 4 && NumThreads >= 8) {
        m[tid] = min(m[tid], m[tid + 4]);
    }
    //__syncthreads();
    
    if(tid < 2 && NumThreads >= 4) {
        m[tid] = min(m[tid], m[tid + 2]);
    }
    //__syncthreads();
    
    if(tid < 1) {
        m[tid] = min(m[tid], m[tid + 1]);
    }
    //__syncthreads();
}
#endif        //  #ifndef CUREDUCEINBLOCK_CUH

