#include "radixsort_implement.h"

namespace onebitsort {
    
template <typename T>
__device__ void reduceInBlock(T * binVertical)
{
    if(threadIdx.x < 128) {
        binVertical[0] += binVertical[0 + 128 *2];
        binVertical[1] += binVertical[1 + 128 *2];
    }
    __syncthreads();
    
    if(threadIdx.x < 64) {
        binVertical[0] += binVertical[0 + 64 *2];
        binVertical[1] += binVertical[1 + 64 *2];
    }
    __syncthreads();
    
    if(threadIdx.x < 32) {
        binVertical[0] += binVertical[0 + 32 *2];
        binVertical[1] += binVertical[1 + 32 *2];
    }
    __syncthreads();
    
    if(threadIdx.x < 16) {
        binVertical[0] += binVertical[0 + 16 *2];
        binVertical[1] += binVertical[1 + 16 *2];
    }
    __syncthreads();
    
    if(threadIdx.x < 8) {
        binVertical[0] += binVertical[0 + 8 *2];
        binVertical[1] += binVertical[1 + 8 *2];
    }
    __syncthreads();
    
    if(threadIdx.x < 4) {
        binVertical[0] += binVertical[0 + 4 *2];
        binVertical[1] += binVertical[1 + 4 *2];
    }
    __syncthreads();
    
    if(threadIdx.x < 2) {
        binVertical[0] += binVertical[0 + 2 *2];
        binVertical[1] += binVertical[1 + 2 *2];
    }
    __syncthreads();
    
    if(threadIdx.x < 1) {
        binVertical[0] += binVertical[0 + 1 *2];
        binVertical[1] += binVertical[1 + 1 *2];
    }
    __syncthreads();
}

template <typename T>
__device__ void scanInBlock(T * sum, T * idata)
{
    int i = threadIdx.x;
// initial value
    sum[i*2] = idata[i*2];
    sum[i*2+1] = idata[i*2+1];
    __syncthreads();
    
// up sweep
    if((i & 1)==1) {
        sum[i*2] = sum[i*2] + sum[(i-1)*2];
        sum[i*2+1] = sum[i*2+1] + sum[(i-1)*2+1];
    }
    __syncthreads();
    
    if((i & 3)==3) {
        sum[i*2] = sum[i*2] + sum[(i-2)*2];
        sum[i*2+1] = sum[i*2+1] + sum[(i-2)*2+1];
    }
    __syncthreads();
    
    if((i & 7)==7) {
        sum[i*2] = sum[i*2] + sum[(i-4)*2];
        sum[i*2+1] = sum[i*2+1] + sum[(i-4)*2+1];
    }
    __syncthreads();
    
    if((i & 15)==15) {
        sum[i*2] = sum[i*2] + sum[(i-8)*2];
        sum[i*2+1] = sum[i*2+1] + sum[(i-8)*2+1];
    }
    __syncthreads();
    
    if((i & 31)==31) {
        sum[i*2] = sum[i*2] + sum[(i-16)*2];
        sum[i*2+1] = sum[i*2+1] + sum[(i-16)*2+1];
    }
    __syncthreads();
    
    if((i & 63)==63) {
        sum[i*2] = sum[i*2] + sum[(i-32)*2];
        sum[i*2+1] = sum[i*2+1] + sum[(i-32)*2+1];
    }
    __syncthreads();
    
    if((i & 127)==127) {
        sum[i*2] = sum[i*2] + sum[(i-64)*2];
        sum[i*2+1] = sum[i*2+1] + sum[(i-64)*2+1];
    }
    __syncthreads();
    
    if((i & 255)==255) {
        sum[i*2] = sum[i*2] + sum[(i-128)*2];
        sum[i*2+1] = sum[i*2+1] + sum[(i-128)*2+1];
    }
    __syncthreads();
    
// down sweep
    T tmp;
    if((i & 255)==255) {
        sum[i*2] = 0;
        sum[i*2+1] = 0;
        
        tmp = sum[(i-128)*2];
        sum[(i-128)*2] = 0;
        sum[i*2] = tmp;
        
        tmp = sum[(i-128)*2+1];
        sum[(i-128)*2+1] = 0;
        sum[i*2+1] = tmp;
    }
    __syncthreads();
    
    if((i & 127)==127) {
        tmp = sum[(i-64)*2];
        sum[(i-64)*2] = sum[i*2];
        sum[i*2] += tmp;
        
        tmp = sum[(i-64)*2+1];
        sum[(i-64)*2+1] = sum[i*2+1];
        sum[i*2+1] += tmp;
    }
    __syncthreads();
    
    if((i & 63)==63) {
        tmp = sum[(i-32)*2];
        sum[(i-32)*2] = sum[i*2];
        sum[i*2] += tmp;
        
        tmp = sum[(i-32)*2+1];
        sum[(i-32)*2+1] = sum[i*2+1];
        sum[i*2+1] += tmp;
    }
    __syncthreads();
    
    if((i & 31)==31) {
        tmp = sum[(i-16)*2];
        sum[(i-16)*2] = sum[i*2];
        sum[i*2] += tmp;
        
        tmp = sum[(i-16)*2+1];
        sum[(i-16)*2+1] = sum[i*2+1];
        sum[i*2+1] += tmp;
    }
    __syncthreads();
    
    if((i & 15)==15) {
        tmp = sum[(i-8)*2];
        sum[(i-8)*2] = sum[i*2];
        sum[i*2] += tmp;
        
        tmp = sum[(i-8)*2+1];
        sum[(i-8)*2+1] = sum[i*2+1];
        sum[i*2+1] += tmp;
    }
    __syncthreads();
    
    if((i & 7)==7) {
        tmp = sum[(i-4)*2];
        sum[(i-4)*2] = sum[i*2];
        sum[i*2] += tmp;
        
        tmp = sum[(i-4)*2+1];
        sum[(i-4)*2+1] = sum[i*2+1];
        sum[i*2+1] += tmp;
    }
    __syncthreads();
    
    if((i & 3)==3) {
        tmp = sum[(i-2)*2];
        sum[(i-2)*2] = sum[i*2];
        sum[i*2] += tmp;
        
        tmp = sum[(i-2)*2+1];
        sum[(i-2)*2+1] = sum[i*2+1];
        sum[i*2+1] += tmp;
    }
    __syncthreads();
    
    if((i & 1)==1) {
        tmp = sum[(i-1)*2];
        sum[(i-1)*2] = sum[i*2];
        sum[i*2] += tmp;
        
        tmp = sum[(i-1)*2+1];
        sum[(i-1)*2+1] = sum[i*2+1];
        sum[i*2+1] += tmp;
    }
    __syncthreads();
}
}
